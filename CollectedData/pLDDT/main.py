import torch
import pandas as pd
import numpy as np
import random

# from dataloader.data import PDBDataLoader
from torch.utils.data import DataLoader
import os
from utils.utils import graph_collate, group_node_rep, target_group, split
import argparse
from calibration.temperature_scaling import ModelWithTemperature
from calibration.std_scaling import STDScaling
from calibration.gemo_scaling import GemoScaling
from utils.visualize import Metric, VisualizeGemo#, LDDTMetric
from dataloader.ae_lddt_3d import AELDDTGEMO
from dataloader.ae_lddt import AELDDT


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    print(args)
    device = torch.device(f"cuda:{args.device}")

    if args.alg == 'std_scaling':
        metric = Metric()

        dataset = AELDDT(select=args.select)

        print(f'successfully load data, length {len(dataset)}...')
        val_set, test_set = split(dataset, frac=args.frac, seed=args.data_seed)
        print(f'successfully split data into val set {len(val_set)} and test set {len(test_set)}...')
        calib_model = STDScaling(device, metric, method=args.method, select=args.select)
        calib_model.set_temeprature(val_set)

        metric.ence_eval_residue(test_set)
        metric.summary(test_set, clamp=True)

        metric.ence_eval_protein(test_set)
        metric.summary_protein(test_set)

    elif args.alg == 'gemo_scaling':
        vis_ae = VisualizeGemo(dev=device, args=args)

        dataset = AELDDTGEMO(select=args.select)

        print(f'successfully load data, length {len(dataset)}...')
        val_set, test_set = split(dataset, frac=args.frac, seed=args.data_seed)
        # val_set, test_set = dataset, dataset

        save_dir = f'test_save/{args.select}_{args.frac}_{args.data_seed}'

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        print(f'successfully split data into val set {len(val_set)} and test set {len(test_set)}...')

        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=graph_collate, drop_last=False)
        
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, collate_fn=graph_collate)
        calib_model = GemoScaling(device, handler=vis_ae, select='ae', epochs=args.epochs, save_dir=save_dir, args=args)
        if not args.model_path:

            calib_model.regress_lddt(val_loader, test_loader)
        vis_ae.set_model([calib_model.graph_construction_model, calib_model.gearnet, calib_model.reg_head])
        if args.model_path:
            vis_ae.load_model(args.model_path)

        vis_ae.summary_plddt(test_loader)


    else:
        NotImplementedError


setup_seed(0)
parser = argparse.ArgumentParser(description='PyTorch implementation of calibration protein')
parser.add_argument('--device', type=int, default=2, help='which gpu to use')
parser.add_argument('--data_seed', type=int, default=1, help='random seed for random sampler')
parser.add_argument('--frac', type=float, default=0.8, help='ratio of select validation samples')
parser.add_argument('--dataset', type=str, default='', help='select dataset')
parser.add_argument('--alg', type=str, default='ts', choices=['ts', 'ts-residues', 'std_scaling', 'gemo_scaling'], help='select calibration method')
parser.add_argument('--batch_size', type=int, default=32, help='batch size. -1 denotes load all samples')
parser.add_argument('--epochs', type=int, default=5, help='batch size. -1 denotes load all samples')
parser.add_argument('--model_path', type=str, default='', help='model path')
parser.add_argument('--select', type=str, default='lddt', help='which metric to be calibrated')
parser.add_argument('--feature', type=int, default='2', help='use which feature')
parser.add_argument('--group', type=str, default='AH_MP', help='use which group for data2')
parser.add_argument('--method', type=str, default='reg', choices=['reg', 'nll'])


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)