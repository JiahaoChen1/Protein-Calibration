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

    def loss_function_ae(self, gt_list, pred_list, substract, temperature):
        first = torch.log(temperature * 2.5066282746310002 * pred_list + 1e-7)
        second = 0.5 * substract / (((temperature * pred_list) ** 2) + 1e-7)
        return (first + second).mean()

    def set_temeprature(self, valid_loader, test_loader):
        self.to(self.dev)
        for e in range(1, self.epochs+1):
            self.train()
            self.avg.reset()
            for idx, (protein, y, py, ae, pae, substract, add_feature) in enumerate(valid_loader):
                # print(protein)
                protein = graph_cuda(protein, self.dev)
                ae, pae = ae.to(self.dev), pae.to(self.dev)
                substract = substract.to(self.dev)
                add_feature = add_feature.to(self.dev)
                py = py.to(self.dev)

                ae, pae = ae.flatten(), pae.flatten()
                mask_indices = (ae != -1)
                ae, pae = ae[mask_indices], pae[mask_indices]
                add_feature = add_feature.flatten()[mask_indices].unsqueeze(1)

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

                residue_feature = self.gearnet(protein, input_feature)['node_feature'] 

                reg_target = self.reg_head(residue_feature)
                substract = substract.flatten()
                substract = substract[mask_indices]

                reg_target = reg_target.flatten()
                if self.args.method == 'reg':
                    loss = ((ae - pae * reg_target) ** 2).mean()
                elif self.args.method == 'nll':
                    loss = self.loss_function_ae(ae, pae, substract, reg_target)

                self.optimizer_reghead.zero_grad()
                self.optimizer_gearnet.zero_grad()
                loss.backward()
                self.optimizer_gearnet.step()
                self.optimizer_reghead.step() 
                # valid_loader.set_postfix(loss=loss.item(), epoch=e)
                self.avg.update(loss.item(), n=32)
            print('*******')
            print(f'Epoch {e} : Loss {self.avg}')

            if e % 100 == 0:

                ckpt = {'graph_construction_model':self.graph_construction_model.state_dict(),
                        'gearnet': self.gearnet.state_dict(),
                        'reg_head': self.reg_head.state_dict()}
                torch.save(ckpt, os.path.join(self.save_dir, f'epoch_{e}.pth'))

                self.gearnet.eval()
                self.reg_head.eval()

                aes, paes, calib_paes = [], [], []
                with torch.no_grad():
                    for protein, _, py, ae, pae, substract, add_feature in test_loader:
                        ae, pae = ae.to(self.dev), pae.to(self.dev)
                        add_feature = add_feature.to(self.dev)
                        add_feature = add_feature.flatten().unsqueeze(1)
                        py = py.reshape(-1, 3).to(self.dev)

                        reg_targets = []
                        split_in_ = 64

                        num_residues = protein['graph'].num_residues
                        split, tail = num_residues // split_in_, num_residues % split_in_
                        i = 0
                        for i in range(split):
                            protein_split = data.Protein.pack([protein['graph'] for _ in range(split_in_)])
                            protein_split = graph_cuda(protein_split, self.dev)
                            protein_split =self.graph_construction_model(protein_split) # 建图
                            if self.args.feature == 1:
                                input_feature = torch.cat([protein_split.node_feature.float(), add_feature[i * split_in_  * num_residues: (i + 1) * split_in_ * num_residues, :]], dim=1)
                            elif self.args.feature == 2:
                                input_feature = torch.cat([py[i * split_in_  * num_residues: (i + 1) * split_in_ * num_residues, :], add_feature[i * split_in_  * num_residues: (i + 1) * split_in_ * num_residues, :]], dim=1)
                            elif self.args.feature == 3:
                                input_feature = torch.cat([protein_split.node_feature.float(), py[i * split_in_  * num_residues: (i + 1) * split_in_ * num_residues, :], add_feature[i * split_in_  * num_residues: (i + 1) * split_in_ * num_residues, :]], dim=1)
                            else:
                                assert 0
                            residue_feature = self.gearnet(protein_split, input_feature)['node_feature'] 
                            reg_target = self.reg_head(residue_feature)
                            reg_targets.append(reg_target)
                            
                        if tail != 0:
                            protein_split = data.Protein.pack([protein['graph'] for _ in range(tail)])
                            protein_split = graph_cuda(protein_split, self.dev)
                            protein_split =self.graph_construction_model(protein_split) 

                            if self.args.feature == 1:
                                input_feature = torch.cat([protein_split.node_feature.float(), add_feature[i * split_in_  * num_residues: (i  * split_in_ + tail)  * num_residues, :]], dim=1)
                            elif self.args.feature == 2:
                                input_feature = torch.cat([py[i * split_in_  * num_residues: (i  * split_in_ + tail)  * num_residues, :], add_feature[i * split_in_  * num_residues: (i  * split_in_ + tail)  * num_residues, :]], dim=1)
                            elif self.args.feature == 3:
                                input_feature = torch.cat([protein_split.node_feature.float(), py[i * split_in_  * num_residues: (i  * split_in_ + tail)  * num_residues, :], add_feature[i * split_in_  * num_residues: (i  * split_in_ + tail)  * num_residues, :]], dim=1)
                            else:
                                assert 0
                            residue_feature = self.gearnet(protein_split, input_feature)['node_feature'] 
                            reg_target = self.reg_head(residue_feature)
                            reg_targets.append(reg_target)
                            reg_target = torch.cat(reg_targets, dim=0)
                        else:
                            reg_target = torch.cat(reg_targets, dim=0)

                        
                        reg_target = reg_target.flatten()

                        ae, pae = ae.flatten(), pae.flatten()

                        aes.append(ae)
                        paes.append(pae)
                        calib_paes.append(pae * reg_target)
                    
                    aes, paes, calib_paes = torch.cat(aes), torch.cat(paes), torch.cat(calib_paes)
                    v = ((aes - paes) ** 2).mean()
                    vb = ((aes - calib_paes) ** 2).mean()
                    print('##########################')
                    print('before calibration...')
                    print(f'RMSE is {v}')
                    print('after calibration...')
                    print(f'RMSE is {vb}')
                    print('##########################')
