import torch
# import matplotlib.pyplot as plt
from utils.utils import graph_cuda, group_node_rep, group_node_indices
from tqdm import tqdm
from calibration.std_scaling import ENCELDDT, ENCE
from torchdrug import core, data
import matplotlib.pyplot as plt


class Metric:
    def __init__(self) -> None:
        self.temperature = 1. 

    def scale1(self, pred):
        return pred * self.temperature / (pred * self.temperature + 1) * 100
    
    def scale2(self, pred):
        return (pred * self.temperature).sigmoid() * 100
    
    def scale3(self, pred):
        return (pred * self.temperature) 
    
    def set_temperature(self, temperature):
        self.temperature = temperature
        print(self.temperature)

    def loss_function(self, gt_list, pred_list, temperature):
        l1 = torch.log(temperature + 1e-7).mean()
        l2 = torch.mean((gt_list ** 2) / (2 * (temperature ** 2) * (pred_list ** 2) + 1e-7))
        loss = l1 + l2
        return loss
    
    def ence_eval(self, loader):
        e = ENCE()
        truths, preds = [], []
        for y, py, gt, pred in loader:
            truths.append(((y - py) ** 2).sum(dim=-1).sqrt().flatten())
            preds.append(pred.flatten())
        truths, preds = torch.cat(truths, dim=0), torch.cat(preds, dim=0)
        temp = e(preds, truths)
        tempb = e(preds * self.temperature, truths)
        print('##########################')
        print('before calibration...')
        print(f'ENCE is {temp}')
        print('after calibration...')
        print(f'ENCE is {tempb}')
        print('##########################')
    
    def summary(self, loader, clamp=False):

        aes, paes, calib_paes = [], [], []
        for _, _, gt, pred in loader:
            aes.append(gt.flatten())
            paes.append(pred.flatten())
            if clamp:
                calib_paes.append(torch.clamp(self.scale3(pred.flatten()), 0, 100))
            else:
                calib_paes.append(self.scale3(pred.flatten()))
                # print(calib_paes[-1])

        aes, paes, calib_paes = torch.cat(aes), torch.cat(paes), torch.cat(calib_paes)

        # p_value, value = calib_paes.long(), aes
        # max_pae = int(torch.max(p_value, dim=0)[0])
        # min_pae = int(torch.min(p_value, dim=0)[0])

        # mean_list = []
        # std_list = []
        # indices_list = []
        # for i in range(min_pae, max_pae + 1):
        #     indices = (p_value == i)
        #     v = value[indices]
        #     print(torch.sum(indices.float()))
        #     indices_list.append(i)
        #     # mm, bottom, high = torch_compute_confidence_interval(v)
            
        #     mean_list.append(float(torch.mean(v)))
        #     std_list.append(float(torch.std(v, unbiased=False)))
        #     # std_list.append(float(torch.std(v)))
        # print(indices_list)
        # print(mean_list)
        # print(std_list)
        # # assert 0

        v = ((aes - paes) ** 2).mean()
        vb = ((aes - calib_paes) ** 2).mean()
        
        print('##########################')
        print('before calibration...')
        print(f'RMSE is {v}')
        print('after calibration...')
        print(f'RMSE is {vb}')
        print('##########################')
    

    def ence_eval_protein(self, loader):
        e = ENCE()
        truths = []
        # ys, pys = [], []

        sigmas, calib_simgas = [], []
        for y, py, gt, pred in loader:
            if pred.shape[0] == 1:
                continue

            # average = (pred[:, None] + pred[None, :]) / 200
            average = (pred / 100)
            calib_pred = torch.clamp(pred * self.temperature, 0, 100)
            # avergae_calib = (calib_pred[:, None] + calib_pred[None, :]) / 200
            avergae_calib = (calib_pred / 100)

            dist_gt = ((y[:, None, :] - y[None, :, :]) ** 2).sum(dim=-1).sqrt()
            dist_pred = ((py[:, None, :] - py[None, :, :]) ** 2).sum(dim=-1).sqrt()

            sigmas.append((-2 / torch.log(1 - average ** 2)).sqrt().mean().unsqueeze(0))
            calib_simgas.append((-2 / torch.log(1 - avergae_calib ** 2)).sqrt().mean().unsqueeze(0))

            distance = ((dist_gt - dist_pred) ** 2).sqrt()
            distance = torch.mean(distance, dim=-1)
            truths.append(distance.mean().unsqueeze(0))

        sigmas = torch.cat(sigmas)
        calib_simgas = torch.cat(calib_simgas)
        truths = torch.cat(truths)
        
        temp, temp2 = e(sigmas, truths)
        tempb, tempb2 = e(calib_simgas, truths)
        print('##########################')
        print('before calibration...')
        print(f'ENCE (protein) is {temp} {temp2}')
        print('after calibration...')
        print(f'ENCE (protein) is {tempb} {tempb2}')
        print('##########################')


    def ence_eval_residue(self, loader):
        e = ENCE()
        truths = []
        # ys, pys = [], []

        sigmas, calib_simgas = [], []
        for y, py, gt, pred in loader:
            if pred.shape[0] == 1:
                continue
            # average = (pred[:, None] + pred[None, :]) / 200
            average = (pred / 100)
            calib_pred = torch.clamp(pred * self.temperature, 0, 100)
            # avergae_calib = (calib_pred[:, None] + calib_pred[None, :]) / 200
            avergae_calib = (calib_pred / 100)

            dist_gt = ((y[:, None, :] - y[None, :, :]) ** 2).sum(dim=-1).sqrt()
            dist_pred = ((py[:, None, :] - py[None, :, :]) ** 2).sum(dim=-1).sqrt()

            sigmas.append((-2 / torch.log(1 - average ** 2)).sqrt())
            calib_simgas.append((-2 / torch.log(1 - avergae_calib ** 2)).sqrt())
            
            distance = ((dist_gt - dist_pred) ** 2).sqrt()
            distance = torch.mean(distance, dim=-1)
            truths.append(distance)


        sigmas = torch.cat(sigmas)
        calib_simgas = torch.cat(calib_simgas)
        truths = torch.cat(truths)
        
        temp, temp2 = e(sigmas, truths)
        tempb, tempb2 = e(calib_simgas, truths)
        print('##########################')
        print('before calibration...')
        print(f'ENCE (residue) is {temp} {temp2}')
        print('after calibration...')
        print(f'ENCE (residue) is {tempb} {tempb2}')
        print('##########################')
    
    def summary_protein(self, loader):
        # rmse = []
        # rmse_calib = []
        aes, paes, calib_paes = [], [], []
        for _, _, gt, pred in loader:
            # temp = ((gt - pred) ** 2).mean()
            # rmse.append(float(temp))
            # temp_calib = ((gt - self.scale3(pred)) ** 2).mean()
            # rmse_calib.append(float(temp_calib))
            aes.append(gt.flatten().mean().unsqueeze(0))
            paes.append(pred.flatten().mean().unsqueeze(0))
            calib_pred = self.scale3(pred.flatten())
            calib_pred = torch.clamp(calib_pred, 0, 100)
            calib_paes.append(calib_pred.mean().unsqueeze(0))

        # v = sum(rmse) / len(rmse)
        # vb = sum(rmse_calib) / len(rmse_calib)
        aes, paes, calib_paes = torch.cat(aes), torch.cat(paes), torch.cat(calib_paes)


        # p_value, value = calib_paes.long(), aes
        # max_pae = int(torch.max(p_value, dim=0)[0])
        # min_pae = int(torch.min(p_value, dim=0)[0])

        # mean_list = []
        # std_list = []
        # indices_list = []
        # for i in range(min_pae, max_pae + 1):
        #     indices = (p_value == i)
        #     v = value[indices]
        #     print(torch.sum(indices.float()))
        #     indices_list.append(i)
        #     # mm, bottom, high = torch_compute_confidence_interval(v)
            
        #     mean_list.append(float(torch.mean(v)))
        #     std_list.append(float(torch.std(v, unbiased=False)))
        #     # std_list.append(float(torch.std(v)))
        # print(indices_list)
        # print(mean_list)
        # print(std_list)
        # assert 0

        v = ((aes - paes) ** 2).mean()
        vb = ((aes - calib_paes) ** 2).mean()
        print('##########################')
        print('before calibration...')
        print(f'RMSE (protein) is {v}')
        print('after calibration...')
        print(f'RMSE (protein) is {vb}')
        print('##########################')



class VisualizeGemo:
    def __init__(self, dev, args) -> None:
        self.gts = []
        self.preds = []
        self.dev = dev
        self.args = args
    
    def set_model(self, model):
        self.graph_construction_model = model[0]
        self.gearnet  = model[1] 
        self.reg_head  = model[2]

        self.graph_construction_model.to(self.dev)
        self.gearnet.to(self.dev)
        self.reg_head.to(self.dev)
    
    def load_model(self, model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        self.gearnet.load_state_dict(ckpt['gearnet'])
        self.reg_head.load_state_dict(ckpt['reg_head'])

    def re_scale3(self, temperature, pred_list):
        return 100 * torch.sqrt(1 - torch.exp(1 / (temperature ** 2) * torch.log(1 - (pred_list / 100) ** 2 + 1e-7)))
    
    def loss_function(self, gt_list, pred_list, temperature):
        # T = gt_list.shape[0]
        l1 = torch.log(temperature + 1e-7).mean()
        l2 = torch.mean((gt_list ** 2) / (2 * (temperature ** 2) * (pred_list ** 2) + 1e-7))
        # print(l1, l2)
        loss = l1 + l2
        # loss = temperature.log().sum() + torch.sum((gt_list ** 2) / (2 * (temperature ** 2) * (pred_list ** 2) + 1e-7))
        return loss
    
    def summary(self, test_loader):
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
                    protein_split =self.graph_construction_model(protein_split) # 建图
                    # input_feature = torch.cat([protein_split.node_feature.float(), add_feature[i * split_in_  * num_residues: (i  * split_in_ + tail)  * num_residues, :]], dim=1)
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
                
                # reg_target = reg_target.repeat(reg_target.shape[0])
                ae, pae = ae.flatten(), pae.flatten()

                # temp =  ((ae - pae) ** 2).mean()
                # rmse.append(float(temp))
                # temp_calib = ((ae - pae * reg_target) ** 2).mean()
                # rmse_calib.append(float(temp_calib))
                aes.append(ae)
                paes.append(pae)
                calib_paes.append(pae * reg_target)
            
            # v = sum(rmse) / len(rmse)
            # vb = sum(rmse_calib) / len(rmse_calib)
            aes, paes, calib_paes = torch.cat(aes), torch.cat(paes), torch.cat(calib_paes)
            v = ((aes - paes) ** 2).mean()
            vb = ((aes - calib_paes) ** 2).mean()
            print('##########################')
            print('before calibration...')
            print(f'RMSE is {v}')
            print('after calibration...')
            print(f'RMSE is {vb}')
            print('##########################')


    def summary_ence(self, test_loader):
        self.gearnet.eval()
        self.reg_head.eval()
        preds, truths = [], []
        old_preds = []
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
                    protein_split =self.graph_construction_model(protein_split) # 建图
                    # input_feature = torch.cat([protein_split.node_feature.float(), add_feature[i * split_in_  * num_residues: (i  * split_in_ + tail)  * num_residues, :]], dim=1)
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
                truths.append(substract.sqrt().flatten())

                preds.append(pae.flatten() * reg_target)
                old_preds.append(pae.flatten())

            e = ENCE()
            
            truths, preds = torch.cat(truths, dim=0), torch.cat(preds, dim=0)
            old_preds = torch.cat(old_preds, dim=0)
            temp = e(old_preds, truths)
            tempb = e(preds, truths)
            print('##########################')
            print('before calibration...')
            print(f'ENCE is {temp}')
            print('after calibration...')
            print(f'ENCE is {tempb}')
            print('##########################')
    
    def summary_plddt(self, test_loader):
        self.gearnet.eval()
        self.reg_head.eval()

        lddts, dist, plddts, calib_plddts = [], [], [], []
        lddts_pro, dist_pro, plddts_pro, calib_plddts_pro = [], [], [], []
        sigmas, sigmas_pro = [], []
        calib_sigmas, calib_sigmas_pro = [], []

        e = ENCE()
        scores = []
        with torch.no_grad():
            for (protein, y, py, lddt, plddt, _, _, key) in test_loader:
                # print(key)
                protein = graph_cuda(protein, self.dev)
                lddt, plddt = lddt.to(self.dev), plddt.to(self.dev)
                
                add_feature = plddt.flatten().unsqueeze(1) 
                y = y.reshape(-1, 3).to(self.dev)
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
                scores.append(reg_target)
                # reg_target = reg_target.repeat(reg_target.shape[0])
                lddt, plddt = lddt.flatten(), plddt.flatten()
                calib_plddt = torch.clamp(plddt * reg_target, 0, 100)
                a = ((y[:, None, :] - y[None, :, :]) ** 2).sum(dim=-1).sqrt()
                b = ((py[:, None, :] - py[None, :, :]) ** 2).sum(dim=-1).sqrt()


                lddts.append(lddt)
                plddts.append(plddt)
                calib_plddts.append(calib_plddt)
                dist.append( torch.mean(((a - b) ** 2).sqrt(), dim=-1))
            
                lddts_pro.append(lddt.mean().unsqueeze(0))
                plddts_pro.append(plddt.mean().unsqueeze(0))
                calib_plddts_pro.append(calib_plddt.mean().unsqueeze(0))
                dist_pro.append(dist[-1].mean().unsqueeze(0))

                sigmas.append((-2 / torch.log(1 - (plddt / 100) ** 2)).sqrt())
                sigmas_pro.append((-2 / torch.log(1 - (plddt / 100) ** 2)).sqrt().mean().unsqueeze(0))

                calib_sigmas.append((-2 / torch.log(1 - (calib_plddt / 100) ** 2)).sqrt())
                calib_sigmas_pro.append((-2 / torch.log(1 - (calib_plddt / 100) ** 2)).sqrt().mean().unsqueeze(0))

                if key[0] == '2BBL':
                    # print(plddt)
                    # print(lddt)
                    # print(calib_plddt)
                    index_b = 1
                    index1 = (torch.where(plddt > 90)[0] + index_b).tolist()
                    index2 = ((torch.where((plddt <= 90) & (plddt > 80))[0]) + index_b).tolist()
                    index3 = ((torch.where((plddt <= 80) & (plddt > 70))[0]) + index_b).tolist()
                    index4 = ((torch.where(plddt <= 70)[0]) + index_b).tolist()
                    print(index1)
                    print(index2)
                    print(index3)
                    print(index4)

                    print('*********')
                    index1 = (torch.where(calib_plddt > 90)[0] + index_b).tolist()
                    index2 = ((torch.where((calib_plddt <= 90) & (calib_plddt > 80))[0]) + index_b).tolist()
                    index3 = ((torch.where((calib_plddt <= 80) & (calib_plddt > 70))[0]) + index_b).tolist()
                    index4 = ((torch.where(calib_plddt <= 70)[0]) + index_b).tolist()
                    print(index1)
                    print(index2)
                    print(index3)
                    print(index4)
            
            assert 0

            lddts, plddts, calib_plddts, dist = torch.cat(lddts), torch.cat(plddts), torch.cat(calib_plddts), torch.cat(dist)
            lddts_pro, plddts_pro, calib_plddts_pro, dist_pro = torch.cat(lddts_pro), torch.cat(plddts_pro), torch.cat(calib_plddts_pro), torch.cat(dist_pro)
            sigmas, sigmas_pro = torch.cat(sigmas), torch.cat(sigmas_pro)
            calib_sigmas, calib_sigmas_pro = torch.cat(calib_sigmas), torch.cat(calib_sigmas_pro)
            
            # p_value, value = calib_plddts.long(), lddts
            # max_pae = int(torch.max(p_value, dim=0)[0])
            # min_pae = int(torch.min(p_value, dim=0)[0])

            # mean_list = []
            # std_list = []
            # indices_list = []
            # for i in range(min_pae, max_pae + 1):
            #     indices = (p_value == i)
            #     v = value[indices]
            #     # print(torch.sum(indices.float()))
            #     indices_list.append(i)
            #     # mm, bottom, high = torch_compute_confidence_interval(v)
                
            #     mean_list.append(float(torch.mean(v)))
            #     std_list.append(float(torch.std(v, unbiased=False)))
            #     # std_list.append(float(torch.std(v)))
            # print(indices_list)
            # print(mean_list)
            # print(std_list)
            # assert 0

            v = ((lddts - plddts) ** 2).mean()
            vb = ((lddts - calib_plddts) ** 2).mean()
            print('##########################')
            print('before calibration...')
            print(f'RMSE is {v}')
            print('after calibration...')
            print(f'RMSE is {vb}')
            print('##########################')

            temp = e(sigmas, dist)
            tempb = e(calib_sigmas, dist)
            print('##########################')
            print('before calibration...')
            print(f'ENCE (residue) is {temp}')
            print('after calibration...')
            print(f'ENCE (residue) is {tempb}')
            print('##########################')

            v = ((lddts_pro - plddts_pro) ** 2).mean()
            vb = ((lddts_pro - calib_plddts_pro) ** 2).mean()
            print('##########################')
            print('before calibration...')
            print(f'RMSE (protein) is {v}')
            print('after calibration...')
            print(f'RMSE (protein) is {vb}')
            print('##########################')

            temp = e(sigmas_pro, dist_pro)
            tempb = e(calib_sigmas_pro, dist_pro)
            print('##########################')
            print('before calibration...')
            print(f'ENCE (protein) is {temp}')
            print('after calibration...')
            print(f'ENCE (protein) is {tempb}')
            print('##########################')


    