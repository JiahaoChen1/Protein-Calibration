import torch
import os
from torch.utils.data import Dataset
# import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from torchdrug import data as ddata
from torchdrug import transforms as dtransforms
from collections.abc import Mapping, Sequence
import random
import pickle
from dataloader.evaluate import canonical_transform


class AELDDTGEMO(Dataset):
    def __init__(self, af_root='./data/af_data/af_data',
                       pdb_root='./pdb_data/data1',
                    #    pairs=None
                       select = 'lddt'
                       ):
        
        pairs = pd.read_csv("./data/pdb_af_pairs_small.csv")
        self.pairs = self.filter_pairs(pairs)
        af_groups = pairs.groupby("uniprot_id")
        self.transform = dtransforms.ProteinView(view='residue') 
        self.collects = {}

        drop = ['A0A0H2YV83', 'P01133', 'P02458', 'P09651', 'P0C0L5', 'P20143',\
                'Q08499', 'Q15788', 'Q5S007', 'Q8R0I0', 'Q8Y565', 'Q96KQ7',\
                'Q9C000', 'Q9H165', 'Q9HCL0', 'Q9V2F4']
        drop = set(drop)
        af_groups = tqdm(af_groups)

        if os.path.exists('data/data_small.pkl'):
            collects_file = open('data/data_small.pkl', 'rb')
            self.collects = pickle.load(collects_file)
            collects_file.close()
        else:
            collects_file = open('data/data_small.pkl', 'wb')
            for (uniprot_id, af_df) in af_groups:
                if uniprot_id in self.collects:
                    assert 0
                if uniprot_id in drop:
                    continue
                
                row = af_df.iloc[af_df["alddt"].argmax()]
                pdb_id = row["pdb_id"]
                asym_id = row["asym_id"]
                af_start = row["uniprot_start"]
                af_end = row["uniprot_end"]
                if row['coverage'] < 0.9:
                    continue
                self.collects[uniprot_id] = {}
                af_pdb_path = os.path.join("./pdb_data/af_pred", "{}_{}_{}.pdb".format(uniprot_id, pdb_id, asym_id))
                # self.af_pdb_paths.append(af_pdb_path)

                af_path = os.path.join(af_root, uniprot_id + ".npz")
                pdb_path = os.path.join(pdb_root, "{}_{}_{}.npz".format(uniprot_id, pdb_id, asym_id))
                
                self.collects[uniprot_id]['af_pdb_path'] = af_pdb_path
                self.collects[uniprot_id]['af_path'] = af_path
                self.collects[uniprot_id]['pdb_path'] = pdb_path
                self.collects[uniprot_id]['fragment_id'] = [int(af_start), int(af_end)]

                protein = ddata.Protein.from_pdb(af_pdb_path)
                self.collects[uniprot_id]['protein'] = protein

            pickle.dump(self.collects, collects_file)
            collects_file.close()

        self.keys = list(self.collects.keys())
        self.is_train = True
        self.select = select
        print(len(self.keys))
    
    def __getitem__(self, item):
        key = self.keys[item]

        gt_file = self.collects[key]['pdb_path']
        # gt_file = gt_file.replace('/home/jiahao_chen/bio/pdb_data', './pdb_data/data1')
        gt_file = np.load(gt_file)

        pre_file = self.collects[key]['af_path']
        # pre_file = pre_file.replace('/home/ning_lin/bio/af_data', './data/af_data/af_data')
        pre_file = np.load(pre_file)

        fragment = self.collects[key]['fragment_id']

        ae = gt_file['ae']
        pae = pre_file['pae'][fragment[0]:fragment[1], fragment[0]:fragment[1]]
        
        lddt = gt_file['lddt']
        plddt = pre_file['plddt'][fragment[0]:fragment[1]]
        
        protein = self.collects[key]['protein'].clone()

        protein = {"graph": protein}
        if self.transform:
            protein = self.transform(protein)
        
        if self.select == 'ae':
            y = canonical_transform(gt_file['pdb_n_xyz_match'], 
                                    gt_file['pdb_ca_xyz_match'],
                                    gt_file['pdb_c_xyz_match'])
            
            py = canonical_transform(pre_file['n_xyz'], 
                                    pre_file['ca_xyz'],
                                    pre_file['c_xyz'])
            # y = gt_file['pdb_ca_xyz_match']
            # py = pre_file['ca_xyz']

            py = py[fragment[0]:fragment[1], fragment[0]:fragment[1], :]

            # add_feature = np.mean(pae, axis=0)
            # add_feature = torch.tensor(add_feature, dtype=torch.float)
            if self.is_train:
                row = random.randint(0, ae.shape[0] - 1)
                ae, pae = ae[row, :], pae[row, :] 
                y, py = y[row, :, :], py[row, :, :]
                y = torch.tensor(y, dtype=torch.float)
                py = torch.tensor(py, dtype=torch.float)
                y, py = y.reshape(-1, 3), py.reshape(-1, 3)
                add_feature = pae
            else:
                y = torch.tensor(y, dtype=torch.float)
                py = torch.tensor(py, dtype=torch.float)
                y, py = y.reshape(-1, 3), py.reshape(-1, 3)
                add_feature = pae
            
            substract = ((y - py ) ** 2).sum(dim=-1)

            return protein, \
                y, \
                py, \
                torch.tensor(ae, dtype=torch.float).flatten(), \
                torch.tensor(pae, dtype=torch.float).flatten(), \
                substract,\
                torch.tensor(add_feature, dtype=torch.float)
        
        elif self.select == 'lddt':
            y = gt_file['pdb_ca_xyz_match']
            py = pre_file['ca_xyz']
            py = py[fragment[0]:fragment[1], :]

            # y = y - y[0, :][None, :]
            # py = py - py[0, :][None, :]
            if self.is_train:
                y = torch.tensor(y, dtype=torch.float)
                py = torch.tensor(py, dtype=torch.float)
                row = torch.randint(0, y.shape[0], (y.shape[0],))
                select_y, select_py = y[row, :], py[row, :]
                dist_gt = ((y - select_y) ** 2).sum(dim=-1).sqrt()
                dist_pred = ((py - select_py) ** 2).sum(dim=-1).sqrt()
            else:
                y = torch.tensor(y, dtype=torch.float)
                py = torch.tensor(py, dtype=torch.float)
                row = torch.randint(0, y.shape[0], (y.shape[0],))
                select_y, select_py = y[row, :], py[row, :]
                dist_gt = ((y - select_y) ** 2).sum(dim=-1).sqrt()
                dist_pred = ((py - select_py) ** 2).sum(dim=-1).sqrt()
            return protein, \
                y,\
                py, \
                torch.tensor(lddt, dtype=torch.float), \
                torch.tensor(plddt, dtype=torch.float), \
                dist_gt, \
                dist_pred,\
                # key,\
                # fragment[0]

    def __len__(self, ):
        return len(self.keys)

    def filter_pairs(self,
                pairs,
                organism=None,
                experimental_method=None,
                is_monomer=None,
                max_length=None,
                min_coverage=None):
        mask = np.ones(len(pairs), dtype=np.bool8)
        if organism:
            mask &= pairs["organism"] == organism
        if experimental_method:
            mask &= pairs["organism"] == experimental_method
        if is_monomer:
            mask &= pairs["is_monomer"] == is_monomer
        if max_length:
            mask &= pairs["length"] <= max_length
        if min_coverage:
            mask &= pairs["coverage"] >= min_coverage

        return pairs[mask]


# if __name__ == '__main__':
#     p = PAE()
#     print(len(p))
    
#     # print(ae, pae)
#     # print(y.shape, py.shape, ae.shape, pae.shape)
#     # print(ae.shape, pae.shape)
#     # print(ae[:, :])
#     # print(pae[:, :])

#     for i in range(len(p)):
#         _, _, ae, pae = p[i]
#         print((ae-pae).sum())


        # print(list(npz_file.keys()))
        # n_xyz, ca_xyz, c_xyz = npz_file['n_xyz'], npz_file['ca_xyz'], npz_file['c_xyz']
        # print(n_xyz.shape)
        # print(ca_xyz.shape)
        # print(c_xyz.shape)
        # trans, rot = make_canonical_transform(n_xyz=n_xyz, ca_xyz=ca_xyz, c_xyz=c_xyz)
        # print(trans.shape, rot.shape)
        # with open(self.pae_files[item], 'r') as f:
        #     pae = json.load(f)[0]
        # print(pae.keys())
        # print(torch.tensor(pae['predicted_aligned_error']).shape)
        # print(type(pae['max_predicted_aligned_error']))
