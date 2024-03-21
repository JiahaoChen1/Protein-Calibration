import torch
import torch.nn as nn
from gearnet import GeometryAwareRelationalGraphNeuralNetwork, RegressionHead
from torchdrug.layers import geometry
from graph_construction import GraphConstructionProtein
from utils.utils import graph_cuda, AverageMeter, group_node_indices
from tqdm import tqdm
from torchdrug import core, data
# from torchdrug import data as ddata
import os
import random
import matplotlib.pyplot as plt
from calibration.std_scaling import ENCE
import numpy as np


class AlphaCarbonNode(nn.Module, core.Configurable):
    """
    Construct only alpha carbon atoms.
    """

    def forward(self, graph):
        """
        Return a subgraph that only consists of alpha carbon nodes.

        Parameters:
            graph (Graph): :math:`n` graph(s)
        """
        mask = (graph.atom_name == data.Protein.atom_name2id["CA"]) & (graph.atom2residue != -1)
        residue2num_atom = graph.atom2residue[mask].bincount(minlength=graph.num_residue)
        residue_mask = residue2num_atom > 0
        mask = mask & residue_mask[graph.atom2residue]
        graph = graph.subgraph(mask).subresidue(residue_mask)
        assert (graph.num_node == graph.num_residue).all()
        # print(graph)
        return graph
    
    
class GemoScaling(nn.Module):
    def __init__(self, dev, handler, select, epochs, save_dir, args) -> None:
        super().__init__()
        self.graph_construction_model = GraphConstructionProtein(node_layers=[AlphaCarbonNode()], 
                                                edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                            geometry.KNNEdge(k=10, min_distance=5),
                                                            geometry.SequentialEdge(max_distance=2)],
                                                edge_feature="gearnet")
        #图模型算法
        # self.gearnet = GeometryAwareRelationalGraphNeuralNetwork(input_dim=21, hidden_dims=[64, 64, 64], 
        #                                                     num_relation=7, edge_input_dim=59, num_angle_bin=8,
        #                                                     batch_norm=True, concat_hidden=True, short_cut=True, readout="sum")
        if args.feature == 1:
            self.gearnet = GeometryAwareRelationalGraphNeuralNetwork(input_dim=22, hidden_dims=[128, 128, 128], 
                                                            num_relation=7, edge_input_dim=None, num_angle_bin=None,
                                                            batch_norm=True, concat_hidden=False, short_cut=True, readout="sum")
        elif args.feature == 2:
            if args.dataset == 'peptide':
                self.gearnet = GeometryAwareRelationalGraphNeuralNetwork(input_dim=4, hidden_dims=[128, 128, 128], 
                                                            num_relation=7, edge_input_dim=None, num_angle_bin=None,
                                                            batch_norm=True, concat_hidden=False, short_cut=True, readout="sum")
            else:
                self.gearnet = GeometryAwareRelationalGraphNeuralNetwork(input_dim=4, hidden_dims=[128, 128, 128], 
                                                                num_relation=7, edge_input_dim=None, num_angle_bin=None,
                                                                batch_norm=True, concat_hidden=False, short_cut=True, readout="sum")
        elif args.feature == 3:
            self.gearnet = GeometryAwareRelationalGraphNeuralNetwork(input_dim=25, hidden_dims=[128, 128, 128], 
                                                            num_relation=7, edge_input_dim=None, num_angle_bin=None,
                                                            batch_norm=True, concat_hidden=False, short_cut=True, readout="sum")
        else:
            assert 0
        if args.dataset == 'peptide':
            self.reg_head = RegressionHead(input_dim=128, output_dim=1)
        else:
            self.reg_head = RegressionHead(input_dim=128, output_dim=1)
        # self.gearnet = GeometryAwareRelationalGraphNeuralNetwork(input_dim=22, hidden_dims=[128, 128, 128], 
        #                                                     num_relation=7, edge_input_dim=None, num_angle_bin=None,
        #                                                     batch_norm=True, concat_hidden=False, short_cut=True, readout="sum")
        #回归，input dim和图模型hidden dims对应

        self.optimizer_gearnet = torch.optim.SGD(self.gearnet.parameters(), lr=1e-4, momentum=0.98)
        self.optimizer_reghead = torch.optim.SGD(self.reg_head.parameters(), lr=1e-4, momentum=0.98)

        self.dev = dev
        self.select = select
        self.epochs = epochs
        self.handler = handler
        self.avg = AverageMeter('loss')
        self.save_dir = save_dir
        self.args = args

        # ckpt = torch.load('/data00/jiahao/bio/test_save2/all_reg/epoch_100.pth', map_location='cpu')
        # self.gearnet.load_state_dict(ckpt['gearnet'])
        # self.reg_head.load_state_dict(ckpt['reg_head'])

    def forward(self, x):
        pass

    def loss_function(self, gt_list, pred_list, temperature):
        l1 = torch.log(temperature + 1e-7).mean()
        l2 = torch.mean((gt_list ** 2) / (2 * (temperature ** 2) * (pred_list ** 2) + 1e-7))
        loss = l1 + l2
        return loss


    def loss_function_lddt(self, plddt, dist_gt, dist_pred, temperature):
        
        plddt_b = torch.clamp(plddt * temperature, 0, 100)
        average_list = plddt_b / 100

        first = -0.5 * torch.log(-torch.log(1 - average_list ** 2 + 1e-12) + 1e-12)
        second = -torch.log(1 - average_list ** 2 + 1e-12) / 4 * ((dist_gt - dist_pred) ** 2)

        return ((first + second) ).mean()

    def regress_lddt(self, valid_loader, test_loader):
        self.to(self.dev)
        for e in range(1, self.epochs+1):
            self.train()
            self.avg.reset()
            for idx, (protein, y, py, lddt, plddt, dist_gt, dist_pred, _) in enumerate(valid_loader):
                # print(protein)
                protein = graph_cuda(protein, self.dev)
                lddt, plddt = lddt.to(self.dev), plddt.to(self.dev)
                y, py = y.to(self.dev), py.to(self.dev)
                dist_gt, dist_pred = dist_gt.to(self.dev), dist_pred.to(self.dev)

                lddt, plddt = lddt.flatten(), plddt.flatten()
                dist_gt, dist_pred = dist_gt.flatten(), dist_pred.flatten()
                mask_indices = (lddt.flatten() != -1)
                lddt, plddt = lddt[mask_indices], plddt[mask_indices]
                dist_gt, dist_pred = dist_gt[mask_indices], dist_pred[mask_indices]
                add_feature = plddt.unsqueeze(1)

                protein = self.graph_construction_model(protein['graph']) # 建图
                py = py.reshape(-1, 3)[mask_indices]
                if self.args.feature == 1:
                    input_feature = torch.cat([protein.node_feature.float(), add_feature], dim=1)
                elif self.args.feature == 2:
                    input_feature = torch.cat([py, add_feature], dim=1)
                elif self.args.feature == 3:
                    input_feature = torch.cat([protein.node_feature.float(), py, add_feature], dim=1)
                else:
                    assert 0
                # input_feature = torch.cat([py, add_feature], dim=1)
                residue_feature = self.gearnet(protein, input_feature)['node_feature'] 

                reg_target = self.reg_head(residue_feature)
                reg_target = reg_target.flatten()
                
                # loss = self.loss_function_lddt(plddt, dist_gt, dist_pred, reg_target)
                if self.args.method == 'reg':
                    loss = ((lddt - plddt * reg_target) ** 2).mean()
                elif self.args.method == 'nll':
                    loss = self.loss_function_lddt(plddt, dist_gt, dist_pred, reg_target)
                    
                self.optimizer_reghead.zero_grad()
                self.optimizer_gearnet.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.gearnet.parameters(), max_norm=1.)
                nn.utils.clip_grad_norm_(self.reg_head.parameters(), max_norm=1.)
                self.optimizer_gearnet.step()
                self.optimizer_reghead.step() 
                # valid_loader.set_postfix(loss=loss.item(), epoch=e)
                self.avg.update(loss.item(), n=32)
            print('*******')
            print(f'Epoch {e} : Loss {self.avg}')

            if e % 100 == 0:
            # if e == self.epochs:

                ckpt = {'graph_construction_model':self.graph_construction_model.state_dict(),
                        'gearnet': self.gearnet.state_dict(),
                        'reg_head': self.reg_head.state_dict()}
                torch.save(ckpt, os.path.join(self.save_dir, f'epoch_{e}.pth'))
                # continue
                # torch.save(ckpt, f'saves/ae_likelihood/epoch_{e}.pth')
                self.gearnet.eval()
                self.reg_head.eval()
                # rmse = []
                # rmse_calib = []
                # scores = []
                lddts, ys, pys, plddts, calib_plddts = [], [], [], [], []
                with torch.no_grad():
                    for (protein, y, py, lddt, plddt, dist_gt, dist_pred, _) in test_loader:

                        protein = graph_cuda(protein, self.dev)
                        lddt, plddt = lddt.to(self.dev), plddt.to(self.dev)
                        
                        add_feature = plddt.flatten().unsqueeze(1)
                        py = py.reshape(-1, 3).to(self.dev)

                        protein = self.graph_construction_model(protein['graph']) # 建图

                        if self.args.feature == 1:
                            input_feature = torch.cat([protein.node_feature.float(), add_feature], dim=1)
                        elif self.args.feature == 2:
                            input_feature = torch.cat([py, add_feature], dim=1)
                        elif self.args.feature == 3:
                            input_feature = torch.cat([protein.node_feature.float(), py, add_feature], dim=1)
                        else:
                            assert 0

                        residue_feature = self.gearnet(protein, input_feature)['node_feature'] 

                        reg_target = self.reg_head(residue_feature)
                        reg_target = reg_target.flatten()
                        # scores.append(reg_target)
                        # reg_target = reg_target.repeat(reg_target.shape[0])
                        lddt, plddt = lddt.flatten(), plddt.flatten()

                        lddts.append(lddt)
                        plddts.append(plddt)
                        calib_plddts.append(plddt * reg_target)
                        ys.append(y)
                        pys.append(pys)
                    
                    
                    lddts, plddts, calib_plddts = torch.cat(lddts), torch.cat(plddts), torch.cat(calib_plddts)
                    # scores = torch.cat(scores)
                    # plt.scatter(torch.arange(scores.shape[0]), scores)
                    # plt.savefig('test.png')
                    v = ((lddts - plddts) ** 2).mean()
                    vb = ((lddts - calib_plddts) ** 2).mean()
                    print('##########################')
                    print('before calibration...')
                    print(f'RMSE is {v}')
                    print('after calibration...')
                    print(f'RMSE is {vb}')
                    print('##########################')
