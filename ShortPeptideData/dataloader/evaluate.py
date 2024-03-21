import numpy as np
import functools
from typing import Tuple

N_ORD = 0
CA_ORD = 1
C_ORD = 2

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

# shape: num_res x 3 x 3
def canonical_transform(
    n_xyz: np.ndarray,
    ca_xyz: np.ndarray,
    c_xyz: np.ndarray) -> np.ndarray:
    translation, rotation = make_canonical_transform(
        n_xyz=n_xyz, ca_xyz=ca_xyz, c_xyz=c_xyz
    )
    # trans_0, rot_0 = translation[0], rotation[0]
    # print("n_xyz: ", rot_0 @ (n_xyz[0] + trans_0))

    # rotation: num_res x 3 x 3
    # translation: num_res x 3
    # ca_xyz: num_res x 3
    canonical_ca_xyz = (rotation[:, None, :, :] @ 
                        (ca_xyz[None, :] + translation[:, None])[:, :, :, None]).squeeze(3)

    return canonical_ca_xyz

# num_res x 3 x 3, num_res x 3 x 3
def get_align_error(
    gt_n_xyz: np.ndarray, gt_ca_xyz: np.ndarray, gt_c_xyz: np.ndarray,
    pred_n_xyz: np.ndarray, pred_ca_xyz: np.ndarray, pred_c_xyz: np.ndarray
    ):
    # shape: num_res(which the structure is aligned) 
    #       x num_res(which the error is predicted)
    #       x 3
    gt_canonical_ca_xyz = canonical_transform(gt_n_xyz, gt_ca_xyz, gt_c_xyz)
    pred_canonical_ca_xyz = canonical_transform(pred_n_xyz, pred_ca_xyz, pred_c_xyz)
    
    align_error = np.round(np.sqrt(np.sum((gt_canonical_ca_xyz - pred_canonical_ca_xyz) ** 2, axis=2)))
    # shape: num_res x num_res
    return align_error


def lddt(predicted_points,
         true_points,
         true_points_mask,
         cutoff=15.,
         per_residue=False):
    """Measure (approximate) lDDT for a batch of coordinates.

    lDDT reference:
    Mariani, V., Biasini, M., Barbato, A. & Schwede, T. lDDT: A local
    superposition-free score for comparing protein structures and models using
    distance difference tests. Bioinformatics 29, 2722–2728 (2013).

    lDDT is a measure of the difference between the true distance matrix and the
    distance matrix of the predicted points.  The difference is computed only on
    points closer than cutoff *in the true structure*.

    This function does not compute the exact lDDT value that the original paper
    describes because it does not include terms for physical feasibility
    (e.g. bond length violations). Therefore this is only an approximate
    lDDT score.

    Args:
        predicted_points: (batch, length, 3) array of predicted 3D points
        true_points: (batch, length, 3) array of true 3D points
        true_points_mask: (batch, length, 1) binary-valued float array.  This mask
        should be 1 for points that exist in the true points.
        cutoff: Maximum distance for a pair of points to be included
        per_residue: If true, return score for each residue.  Note that the overall
        lDDT is not exactly the mean of the per_residue lDDT's because some
        residues have more contacts than others.

    Returns:
        An (approximate, see above) lDDT score in the range 0-1.
    """

    assert len(predicted_points.shape) == 3
    assert predicted_points.shape[-1] == 3
    assert true_points_mask.shape[-1] == 1
    assert len(true_points_mask.shape) == 3

    # Compute true and predicted distance matrices.
    dmat_true = np.sqrt(1e-10 + np.sum(
        (true_points[:, :, None] - true_points[:, None, :])**2, axis=-1))

    dmat_predicted = np.sqrt(1e-10 + np.sum(
        (predicted_points[:, :, None] - predicted_points[:, None, :])**2, axis=-1))
    
    dists_to_score = (
        (dmat_true < cutoff).astype(np.float32) * true_points_mask *
        np.transpose(true_points_mask, [0, 2, 1]) *
        (1. - np.eye(dmat_true.shape[1]))  # Exclude self-interaction.
    )

    # Shift unscored distances to be far away.
    dist_l1 = np.abs(dmat_true - dmat_predicted)

    # True lDDT uses a number of fixed bins.
    # We ignore the physical plausibility correction to lDDT, though.
    score = 0.25 * ((dist_l1 < 0.5).astype(np.float32) +
                    (dist_l1 < 1.0).astype(np.float32) +
                    (dist_l1 < 2.0).astype(np.float32) +
                    (dist_l1 < 4.0).astype(np.float32))

    # Normalize over the appropriate axes.
    reduce_axes = (-1,) if per_residue else (-2, -1)
    norm = 1. / (1e-10 + np.sum(dists_to_score, axis=reduce_axes))
    score = norm * (1e-10 + np.sum(dists_to_score * score, axis=reduce_axes))

    return score

def get_lddt(gt_ca_xyz: np.ndarray, pred_ca_xyz: np.ndarray):
    num_res, _ = gt_ca_xyz.shape
    mask = np.ones((1, num_res, 1))

    return 100 * lddt(pred_ca_xyz[None, :, :], gt_ca_xyz[None, :, :],
                    mask, cutoff=15., per_residue=True).squeeze(0)
