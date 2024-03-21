import torch
import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm

from torchdrug import data as ddata
from torchdrug import transforms as dtransforms
from collections.abc import Mapping, Sequence
import random
import pickle


'''
AH_MP, AH_SL, BHPIN, DSRP, MIX_MP, MIX_SL
'''
class Peptide(Dataset):
    def __init__(self, pdb_data='pdb_data/data2', 
                       group_id='all',
                       ):
        assert group_id in ['AH_MP', 'AH_SL', 'BHPIN', 'DSRP', 'MIX_MP', 'MIX_SL', 'all']
        pairs = pd.read_csv("./data/peptide.csv")
        self.transform = dtransforms.ProteinView(view='residue') 
        self.collects = {}
        if os.path.exists(f'data/peptide/peptide_{group_id}.pkl'):
            collects_file = open(f'data/peptide/peptide_{group_id}.pkl', 'rb')
            self.collects = pickle.load(collects_file)
            collects_file.close()
        else:
            collects_file = open(f'data/peptide/peptide_{group_id}.pkl', 'wb')
            for index, row in pairs.iterrows():
                if row['group_id'] == group_id or group_id == 'all':
                    uniprot_id = row['id']

                    self.collects[uniprot_id] = {}

                    pred_path = row['pred_path']
                    af_pdb_path = os.path.join(pdb_data, "{}".format(pred_path))
                    protein = ddata.Protein.from_pdb(af_pdb_path)
                    self.collects[uniprot_id]['protein'] = protein

                    self.collects[uniprot_id]['lddt'] = row['lddt']
                    self.collects[uniprot_id]['plddt'] = row['plddt']
                    self.collects[uniprot_id]['pred_ca_xyz'] = row['pred_ca_xyz']
                    self.collects[uniprot_id]['target_ca_xyz'] = row['target_ca_xyz']
            pickle.dump(self.collects, collects_file)
            collects_file.close()

        self.keys = list(self.collects.keys())
        self.is_train = True
        print(len(self.keys))
    
    def __getitem__(self, index):
        key = self.keys[index]
        lddt = self.collects[key]['lddt']
        plddt = self.collects[key]['plddt']

        protein = self.collects[key]['protein']
        protein = {"graph": protein}
        if self.transform:
            protein = self.transform(protein)

        lddt = torch.tensor(eval(lddt))
        plddt = torch.tensor(eval(plddt))
        y, py = self.collects[key]['target_ca_xyz'], self.collects[key]['pred_ca_xyz']
        y, py = torch.tensor(eval(y)), torch.tensor(eval(py))
        # if self.is_train:
        #     row = torch.randint(0, y.shape[0], (y.shape[0],))
        #     select_y, select_py = y[row, :], py[row, :]
        #     dist_gt = ((y - select_y) ** 2).sum(dim=-1).sqrt()
        #     dist_pred = ((py - select_py) ** 2).sum(dim=-1).sqrt()
        # else:
        #     row = torch.randint(0, y.shape[0], (y.shape[0],))
        #     select_y, select_py = y[row, :], py[row, :]
        #     dist_gt = ((y - select_y) ** 2).sum(dim=-1).sqrt()
        #     dist_pred = ((py - select_py) ** 2).sum(dim=-1).sqrt()
        return y, py, lddt, plddt


    def __len__(self):
        return len(self.keys)

if __name__ == '__main__':
    d = Peptide()
    print(len(d))
    a = 0
    for i in range(len(d)):
        _, _, lddt, plddt = d[i]
        print((lddt - plddt).sum())
        a = a + (lddt - plddt).sum()
        print('**')
        print(a)