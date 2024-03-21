from collections.abc import Mapping, Sequence
import numpy as np
from typing import Tuple
import torch
import matplotlib.pyplot as plt
from torchdrug import data as ddata
import copy
import time


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    

def graph_collate(batch):
    """
    Convert any list of same nested container into a container of tensors.

    For instances of :class:`data.Graph <torchdrug.data.Graph>`, they are collated
    by :meth:`data.Graph.pack <torchdrug.data.Graph.pack>`.

    Parameters:
        batch (list): list of samples with the same nested container
    """
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            if len(batch[0].shape) == 1:
                max_len = max([x.shape[-1] for x in batch])
                max_numel = max([x.numel() for x in batch])
                numel = max_numel * len(batch)
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
                new_batch = [torch.nn.functional.pad(x, (0, max_len - x.shape[-1]), 'constant', -1) 
                            for x in batch]
            elif len(batch[0].shape) == 2:
                max_len = max([x.shape[-2] for x in batch])
                max_numel = max([x.numel() for x in batch])
                numel = max_numel * len(batch)
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
                new_batch = [torch.nn.functional.pad(x, (0, 0, max_len - x.shape[-2], 0), 'constant', -1)
                            for x in batch]
        else:
            # max_len = max([x.shape[-2] for x in batch])
            # print(max_len)
            if len(batch[0].shape) == 1:
                max_len = max([x.shape[-1] for x in batch])
                new_batch = [torch.nn.functional.pad(x, (0, max_len - x.shape[-1]), 'constant', -1) 
                            for x in batch]
            elif len(batch[0].shape) == 2:
                new_batch = [torch.nn.functional.pad(x, (0, max_len - x.shape[-2], 0, 0), 'constant', -1) 
                            for x in batch]
            
        return torch.stack(new_batch, 0, out=out)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, ddata.Graph):
        return elem.pack(batch)
    elif isinstance(elem, Mapping):
        return {key: graph_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            print()
            raise RuntimeError('Each element in list of batch should be of equal size')
        return [graph_collate(samples) for samples in zip(*batch)]

    raise TypeError("Can't collate data with type `%s`" % type(elem))

def graph_cuda(obj, *args, **kwargs):
    """
    Transfer any nested conatiner of tensors to CUDA.
    """
    if hasattr(obj, "cuda"):
        return obj.cuda(*args, **kwargs)
    elif isinstance(obj, Mapping):
        return type(obj)({k: graph_cuda(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, Sequence):
        return type(obj)(graph_cuda(x, *args, **kwargs) for x in obj)

    raise TypeError("Can't transfer object type `%s`" % type(obj))


def group_node_rep(node_rep, batch_index, frag):
    batch_size = int(max(batch_index)) + 1
    group = []
    count = 0
    for i in range(batch_size):
        num = torch.sum(batch_index == i)
        batch_rep = node_rep[count:count + num]
        batch_rep = batch_rep[frag[i, 0]:frag[i, 1]]
        group.append(batch_rep)
        count += num
    return group

def group_node_indices(batch_index, frag):
    batch_size = int(torch.max(batch_index, dim=0)[0]) + 1
    count = 0
    indices = torch.zeros_like(batch_index)
    for i in range(batch_size):
        num = torch.sum(batch_index == i)
        indices[count:count+num][frag[i, 0]:frag[i, 1]] = 1
        count += num
    return indices

def target_group(target):
    group = []
    for i in range(target.shape[0]):
        t = target[i]
        group.append(t[t != -1])
    return group


def make_canonical_transform(
    n_xyz: np.ndarray,
    ca_xyz: np.ndarray,
    c_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns translation and rotation matrices to canonicalize residue atoms.

    Note that this method does not take care of symmetries. If you provide the
    atom positions in the non-standard way, the N atom will end up not at
    [-0.527250, 1.359329, 0.0] but instead at [-0.527250, -1.359329, 0.0]. You
    need to take care of such cases in your code.

    Args:
        n_xyz: An array of shape [batch, 3] of nitrogen xyz coordinates.
        ca_xyz: An array of shape [batch, 3] of carbon alpha xyz coordinates.
        c_xyz: An array of shape [batch, 3] of carbon xyz coordinates.

    Returns:
        A tuple (translation, rotation) where:
        translation is an array of shape [batch, 3] defining the translation.
        rotation is an array of shape [batch, 3, 3] defining the rotation.
        After applying the translation and rotation to all atoms in a residue:
        * All atoms will be shifted so that CA is at the origin,
        * All atoms will be rotated so that C is at the x-axis,
        * All atoms will be shifted so that N is in the xy plane.
    """
    assert len(n_xyz.shape) == 2, n_xyz.shape
    assert n_xyz.shape[-1] == 3, n_xyz.shape
    assert n_xyz.shape == ca_xyz.shape == c_xyz.shape, (
        n_xyz.shape, ca_xyz.shape, c_xyz.shape)

    # Place CA at the origin.
    translation = -ca_xyz
    n_xyz = n_xyz + translation
    c_xyz = c_xyz + translation

    # Place C on the x-axis.
    c_x, c_y, c_z = [c_xyz[:, i] for i in range(3)]
    # Rotate by angle c1 in the x-y plane (around the z-axis).
    sin_c1 = -c_y / np.sqrt(1e-20 + c_x**2 + c_y**2)
    cos_c1 = c_x / np.sqrt(1e-20 + c_x**2 + c_y**2)
    zeros = np.zeros_like(sin_c1)
    ones = np.ones_like(sin_c1)
    # pylint: disable=bad-whitespace
    c1_rot_matrix = np.stack([np.array([cos_c1, -sin_c1, zeros]),
                                np.array([sin_c1,  cos_c1, zeros]),
                                np.array([zeros,    zeros,  ones])])

    # Rotate by angle c2 in the x-z plane (around the y-axis).
    sin_c2 = c_z / np.sqrt(1e-20 + c_x**2 + c_y**2 + c_z**2)
    cos_c2 = np.sqrt(c_x**2 + c_y**2) / np.sqrt(
        1e-20 + c_x**2 + c_y**2 + c_z**2)
    c2_rot_matrix = np.stack([np.array([cos_c2,  zeros, sin_c2]),
                                np.array([zeros,    ones,  zeros]),
                                np.array([-sin_c2, zeros, cos_c2])])

    # c_rot_matrix = _multiply(c2_rot_matrix, c1_rot_matrix)
    c_rot_matrix = np.einsum("ijb, jkb -> ikb", c2_rot_matrix, c1_rot_matrix)
    # n_xyz = np.stack(apply_rot_to_vec(c_rot_matrix, n_xyz, unstack=True)).T
    n_xyz = np.einsum("ijb, bj -> bi", c_rot_matrix, n_xyz)

    # Place N in the x-y plane.
    _, n_y, n_z = [n_xyz[:, i] for i in range(3)]
    # Rotate by angle alpha in the y-z plane (around the x-axis).
    sin_n = -n_z / np.sqrt(1e-20 + n_y**2 + n_z**2)
    cos_n = n_y / np.sqrt(1e-20 + n_y**2 + n_z**2)
    n_rot_matrix = np.stack([np.array([ones,  zeros,  zeros]),
                                np.array([zeros, cos_n, -sin_n]),
                                np.array([zeros, sin_n,  cos_n])])
    # pylint: enable=bad-whitespace

    rotation = np.einsum("ijb, jkb -> ikb", n_rot_matrix, c_rot_matrix)
    rotation = np.transpose(rotation, [2, 0, 1])
    return translation, rotation


def split(dataset, frac=0.8, seed=1):
    '''
    
    '''
    if seed == -1:
        pass
    else:
        np.random.seed(seed)

    dataset_test = copy.deepcopy(dataset)
    validation_len = int(len(dataset) * frac)
    indices = np.random.choice(len(dataset), validation_len, replace=False)
    validation = torch.utils.data.Subset(dataset, indices)
    dataset_test.is_train = False
    test = torch.utils.data.Subset(dataset_test, list(set([i for i in range(len(dataset))]) - set(indices)))
    return validation, test


def reliability_diagram(means, rmses, begin, end):
    means, rmses = means.numpy(), rmses.numpy()
    x = torch.arange(1, 7, 0.2)

    fig, ax = plt.subplots()
    ax.plot(x, x, '--',color='gray')
    ax.plot(means, rmses, '-o' , color='red')
    # plt.plot(x, rmses, '-o', label='y=x^2', color='green', )

    # ax.patch.set_facecolor('lightskyblue')
    # ax.patch.set_alpha(0.3)
    plt.xticks(np.arange(1, 7, step=1))
    plt.yticks(np.arange(1, 28, step=5))
    # plt.axis('square')
    # Add labels and title
    plt.xlabel('mVAR')
    plt.ylabel('RMSE')
    
    # fig.set_facecolor('lightgray')
    # plt.title('Plot of Lines')

    # Add legend
    # plt.legend()
    # plt.grid(color='w')
    # Display the plot
    plt.savefig('pae.png')


def reliability_diagram_bar(means, rmses, begin, end):
    """
    outputs - a torch tensor (size n x num_classes) with the outputs from the final linear layer
    - NOT the softmaxes
    labels - a torch tensor (size n) with the labels
    """
    plt.clf()
    print(begin, end)
    space = (end - begin)
    n_bins=len(means)
    bin_scores, bin_corrects = means, rmses
    
    # # Reliability diagram
    # bins = torch.linspace(0, space, n_bins + 1)
    width = space / n_bins 
    bin_centers = np.linspace(begin, end - width, n_bins) + width / 2

    # bin_indices = [confidences.ge(bin_lower) * confidences.lt(bin_upper) for bin_lower, bin_upper in zip(bins[:-1], bins[1:])]
    
    # bin_corrects = np.array([ torch.mean(accuracies[bin_index].float()) for bin_index in bin_indices])
    # bin_scores = np.array([ torch.mean(confidences[bin_index].float()) for bin_index in bin_indices])
    # bin_corrects = np.nan_to_num(bin_corrects)
    # bin_scores = np.nan_to_num(bin_scores)
    
    # gap_bin_scores = torch.array()
    # plt.figure(0, figsize=(4, 3))
    gap = np.array(torch.tensor(bin_scores) - bin_corrects)
    
    confs = plt.bar(bin_centers, bin_scores , color=[0, 0, 1], width=width, ec='black')
    bin_corrects = np.nan_to_num(np.array([bin_correct  for bin_correct in bin_corrects]))
    gaps = plt.bar(bin_centers, gap, bottom=bin_corrects, color=[1., 0.7, 0.7], alpha=0.6, width=width, hatch='//', edgecolor='r')
    
    plt.plot([begin, end], [begin, end], '--', color='gray')
    plt.legend([confs, gaps], ['Expected RMSE', 'Gap'], loc='upper left', fontsize='large')

    # Clean up
    # bbox_props = dict(boxstyle="square", fc="lightgrey", ec="gray", lw=1.5)
    # plt.text(0.17, 0.82, "ECE: {:.4f}".format(ece), ha="center", va="center", size=20, weight = 'normal', bbox=bbox_props)

    # plt.title("Reliability Diagram", size=22)
    plt.ylabel("RMSE",  size=14)
    plt.xlabel("mVAR",  size=14)
    plt.xlim(begin,end)
    plt.ylim(0,max(max(rmses), max(means)) + 1)
    plt.savefig(f'reliability_diagram{time.time()}.png')
    # plt.show()