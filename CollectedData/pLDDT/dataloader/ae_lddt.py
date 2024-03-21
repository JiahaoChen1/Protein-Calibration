import torch
import os
from torch.utils.data import Dataset
# import json
import numpy as np
import sys
sys.path.append('/data00/jiahao/bio')
from dataloader.evaluate import canonical_transform
import pandas as pd
from tqdm import tqdm


class AELDDT(Dataset):
    def __init__(self, af_root='./data/af_data/af_data',
                       pdb_root='./pdb_data/data1',
                       select = 'ae'
                       ):
        
        self.select = select
        pairs = pd.read_csv("./data/pdb_af_pairs_small.csv")
        self.pairs = self.filter_pairs(pairs)
        af_groups = pairs.groupby("uniprot_id")

        self.collects = {}

        drop = ['A0A0H2YV83', 'P01133', 'P02458', 'P09651', 'P0C0L5', 'P20143',\
                'Q08499', 'Q15788', 'Q5S007', 'Q8R0I0', 'Q8Y565', 'Q96KQ7',\
                'Q9C000', 'Q9H165', 'Q9HCL0', 'Q9V2F4']
        drop = set(drop)

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

            af_pdb_path = os.path.join(af_root, "{}.pdb".format(uniprot_id))
            # self.af_pdb_paths.append(af_pdb_path)

            af_path = os.path.join(af_root, uniprot_id + ".npz")
            pdb_path = os.path.join(pdb_root, "{}_{}_{}.npz".format(uniprot_id, pdb_id, asym_id))
            
            self.collects[uniprot_id]['af_pdb_path'] = af_pdb_path
            self.collects[uniprot_id]['af_path'] = af_path
            self.collects[uniprot_id]['pdb_path'] = pdb_path
            self.collects[uniprot_id]['fragment_id'] = [int(af_start), int(af_end)]

        self.keys = list(self.collects.keys())
    
    def __getitem__(self, item):
        key = self.keys[item]

        gt_file = self.collects[key]['pdb_path']
        gt_file = np.load(gt_file)

        pre_file = self.collects[key]['af_path']
        pre_file = np.load(pre_file)

        fragment = self.collects[key]['fragment_id']
        # print(fragment)

        ae = gt_file['ae']
        pae = pre_file['pae'][fragment[0]:fragment[1], fragment[0]:fragment[1]]
        
        lddt = gt_file['lddt']
        plddt = pre_file['plddt'][fragment[0]:fragment[1]]
        if self.select == 'ae':
            y = canonical_transform(gt_file['pdb_n_xyz_match'], 
                                    gt_file['pdb_ca_xyz_match'],
                                    gt_file['pdb_c_xyz_match'])
            
            py = canonical_transform(pre_file['n_xyz'], 
                                    pre_file['ca_xyz'],
                                    pre_file['c_xyz'])

            py = py[fragment[0]:fragment[1], fragment[0]:fragment[1], :]
        else:
            y = gt_file['pdb_ca_xyz_match']
            py = pre_file['ca_xyz']
            py = py[fragment[0]:fragment[1], :]

        y = torch.tensor(y, dtype=torch.float)
        py = torch.tensor(py, dtype=torch.float)
        if self.select == 'ae':
            return y,\
                    py,\
                    torch.tensor(ae, dtype=torch.float), \
                    torch.tensor(pae, dtype=torch.float), \
                    # self.keys[item],
                    # fragment
        elif self.select == 'lddt':
            return y,\
                    py,\
                    torch.tensor(lddt, dtype=torch.float), \
                    torch.tensor(plddt, dtype=torch.float), \
                    # self.keys[item]
                    # fragment
        else:
            NotImplementedError

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


if __name__ == '__main__':
    p = AELDDT(select='lddt')
    print(len(p))
    a = 0
    for i in range(len(p)):
        _, _, ae, pae,_,_ = p[i]

        # print((pae - ae).sum())
        index = 9
        a = a + (pae - ae).sum()
        print(a)


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
