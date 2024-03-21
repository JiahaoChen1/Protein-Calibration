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
        

        self.optimizer_gearnet = torch.optim.SGD(self.gearnet.parameters(), lr=1e-4, momentum=0.98)
        self.optimizer_reghead = torch.optim.SGD(self.reg_head.parameters(), lr=1e-4, momentum=0.98)

        self.dev = dev
        self.select = select
        self.epochs = epochs
        self.handler = handler
        self.avg = AverageMeter('loss')
        self.save_dir = save_dir
        self.args = args


    def forward(self, x):
        pass

    def loss_function(self, gt_list, pred_list, temperature):
        l1 = torch.log(temperature + 1e-7).mean()
        l2 = torch.mean((gt_list ** 2) / (2 * (temperature ** 2) * (pred_list ** 2) + 1e-7))
        loss = l1 + l2
        return loss
    
    def loss_function_ae(self, gt_list, pred_list, substract, temperature):
        first = torch.log(temperature * 2.5066282746310002 * pred_list + 1e-7)
        second = 0.5 * substract / (((temperature * pred_list) ** 2) + 1e-7)
        return (first + second).mean()
        # return ((temperature * pred_list - gt_list) ** 2).mean()

    def loss_function_lddt(self, plddt, dist_gt, dist_pred, temperature):
        
        plddt_b = torch.clamp(plddt * temperature, 0, 100)
        average_list = plddt_b / 100

        first = -0.5 * torch.log(-torch.log(1 - average_list ** 2 + 1e-12) + 1e-12)
        second = -torch.log(1 - average_list ** 2 + 1e-12) / 4 * ((dist_gt - dist_pred) ** 2)
        # with torch.no_grad():
        # p = -0.5 * torch.log(-torch.log(1 - average_list ** 2 + 1e-12)) + 1.2655 - torch.log(1 - average_list ** 2 + 1e-12) / 4 * ((dist_gt - dist_pred) ** 2)

        # return (((torch.exp(-1 * p) + 1e-7) ** 0.15)  * (first + second) ).mean()
        return ((first + second) ).mean()

    def re_scale3(self, temperature, pred_list):
        return 100 * torch.sqrt(1 - torch.exp(1 / (temperature ** 2) * torch.log(1 - (pred_list / 100) ** 2 + 1e-7)))
    
    def lddt_alternative3(self, y, py, gt_list, pred_list, temperature):
        prefix = 1 / (torch.pow(temperature, 3) + 1e-7) * torch.pow( -torch.log(1 - (pred_list / 100) ** 2 + 1e-7) / (4 * 3.1415926), 3/2)
        sigma = -torch.log(1 - (pred_list / 100) ** 2 + 1e-7) / (2 * (pred_list ** 2))
        loss = prefix * torch.exp(-0.5 * ((y - py) ** 2).sum(dim=-1) * (sigma ** 2))
        return -torch.log(loss + 1e-7).mean()
    
    def re_scale2(self, temperature, pred_list):
        return 100 * (temperature * pred_list).sigmoid()
    
    def lddt_alternative2(self, gt_list, pred_list, temperature):
        sigma = -torch.log(1 - ((temperature * pred_list).sigmoid()) ** 2 + 1e-7) / 2
        loss = torch.pow(sigma / (2 * 3.1415926), 3) * torch.exp(-0.5 * ((gt_list / 100) ** 2) * (sigma ** 2))
        return -loss.mean()
    
    def re_scale1(self, temperature, pred_list):
        return 100 * (temperature * pred_list / (temperature * pred_list + 1))
    
    def lddt_alternative1(self, gt_list, pred_list, temperature):
        sigma = -torch.log(1 - (temperature * pred_list / (temperature * pred_list + 1)) ** 2 + 1e-7) / 2
        loss = torch.pow(sigma / (2 * 3.1415926), 3) * torch.exp(-0.5 * ((gt_list / 100) ** 2) * (sigma ** 2))
        return -loss.mean()

    def lddt_alternative4(self, y, py, gt_list, pred_list, temperature):
        add_lddt = (pred_list[:, :, None] + pred_list[:, None, :]) / 200
        dis = ((y[:, :, None, :] - y[:, None, :, :]) ** 2).sum(dim=-1)
        mask = torch.ones(add_lddt.shape[-1], add_lddt.shape[-1]).to(self.device)
        mask = torch.triu(mask, 1)
        

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
                
                if self.args.method == 'nll':
                    loss = self.loss_function_lddt(plddt, dist_gt, dist_pred, reg_target)
                elif self.args.method == 'reg':
                    loss = ((lddt - plddt * reg_target) ** 2).mean()

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
