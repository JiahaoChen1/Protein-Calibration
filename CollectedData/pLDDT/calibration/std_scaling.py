from typing import Any
import torch
import torch.nn as nn
# from torch.distributions import MultivariateNormal
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

from utils.utils import reliability_diagram, reliability_diagram_bar
# from scipy.stats import multivariate_normal
import matplotlib.pylab as plt


class ENCE:
    def __init__(self, num_bins=10, select='lddt') -> None:
        self.num_bins = num_bins
        self.intervals_mvar = torch.tensor([0. for _ in range(num_bins)])
        self.intervals_rmse = torch.tensor([0. for _ in range(num_bins)])
        self.num_samples = torch.tensor([0. for _ in range(num_bins)])
        self.select = select

    def __call__(self, sigmas, truths) -> Any:
        num_samples = sigmas.shape[0]
        with torch.no_grad():
            ids = torch.argsort(sigmas, dim=0)
            sigmas = sigmas[ids]
            truths = truths[ids]

            num = num_samples / self.num_bins

            begin = sigmas[0]
            end = sigmas[-1]
            interval = (end - begin) / self.num_bins
            left = [(begin + i * interval) ** 2 for i in range(self.num_bins)]
            left.append((begin + self.num_bins * interval + 1) ** 2)

            sigmas, truths = sigmas ** 2, truths ** 2

            for i in range(self.num_bins):
                # select_begin, select_end = int(i * num), int((i + 1) * num)
                # if i == self.num_bins - 1:
                #     select_end = num_samples
                # sigmas_in_bin = sigmas[select_begin:select_end]
                # truths_in_bin = truths[select_begin:select_end]

                sigmas_in_bin = sigmas[(sigmas >= left[i]) & (sigmas < left[i+1] )]
                truths_in_bin = truths[(sigmas >= left[i] ) & (sigmas < left[i+1] )]
                if i == self.num_bins - 1:
                    sigmas_in_bin = sigmas[(sigmas >= left[i]) & (sigmas <= left[i+1] )]
                    truths_in_bin = truths[(sigmas >= left[i] ) & (sigmas <= left[i+1] )]
                    # print(sum((sigmas >= left[i] ) & (sigmas <= left[i+1] )))
                print(((sigmas >= left[i]) & (sigmas < left[i+1])).sum())
                mvar = torch.mean(sigmas_in_bin).sqrt()
                rmse = torch.mean(truths_in_bin).sqrt()

                self.intervals_mvar[i] = mvar
                self.intervals_rmse[i] = rmse
                # self.num_samples[i] = len(sigmas_in_bin)

            # abs_val = (mvar - rmse).abs() / mvar
        print(self.intervals_mvar)
        print(self.intervals_rmse)
        
        nan_mask = ~torch.isnan(self.intervals_mvar)
        cal_mvar = self.intervals_mvar[nan_mask]
        cal_rmse = self.intervals_rmse[nan_mask]

        abs_val = (cal_mvar- cal_rmse).abs() / cal_mvar #* (self.num_samples / self.num_samples.sum()) * self.num_bins
        abs_val2 = (cal_mvar - cal_rmse).abs()
        # reliability_diagram(self.intervals_mvar, self.intervals_rmse, int(left[0] ** 0.5) , int(left[-1] ** 0.5) )
        reliability_diagram_bar(self.intervals_mvar, self.intervals_rmse, float(begin) , float(end) )
        # reliability_diagram(cal_mvar, cal_rmse, 0, 23)
        # print(abs_val.mean())
        return abs_val.sum() / self.num_bins, abs_val2.sum() / self.num_bins


class ENCELDDT(nn.Module):
    def __init__(self, num_bins=10, select='lddt') -> None:
        self.num_bins = num_bins
        self.intervals_mvar = torch.tensor([0. for _ in range(num_bins)])
        self.intervals_rmse = torch.tensor([0. for _ in range(num_bins)])
        self.select = select

    def __call__(self, sigmas, y, py) -> Any:

        # num_samples = sigmas.shape[0]
        average = (sigmas[:, None] + sigmas[None, :]) / 200
        dist_gt = ((y[:, None, :] - y[None, :, :]) ** 2).sum(dim=-1).sqrt()
        dist_pred = ((py[:, None, :] - py[None, :, :]) ** 2).sum(dim=-1).sqrt()

        dim_ = average.shape[0]
        average = average.flatten()[1:].view(dim_-1, dim_+1)[:,:-1].reshape(dim_, dim_-1).flatten()
        dist_gt = dist_gt.flatten()[1:].view(dim_-1, dim_+1)[:,:-1].reshape(dim_, dim_-1).flatten()
        dist_pred = dist_pred.flatten()[1:].view(dim_-1, dim_+1)[:,:-1].reshape(dim_, dim_-1).flatten()
        truths = dist_gt - dist_pred

        sigmas = (-2 / torch.log(1 - average ** 2)).sqrt()

        # sigmas, truths = truths, sigmas
        with torch.no_grad():
            ids = torch.argsort(sigmas, dim=0)
            sigmas = torch.pow(sigmas[ids], 2)
            truths = torch.pow(truths[ids], 2)
            # num = num_samples / self.num_bins

            begin = sigmas[0]
            end = sigmas[-1]
            interval = (end - begin) / self.num_bins
            left = [begin + i * interval for i in range(self.num_bins)]
            left.append(begin + self.num_bins * interval + 1)

            for i in range(self.num_bins):
            
                sigmas_in_bin = sigmas[(sigmas >= left[i]) & (sigmas < left[i+1] )]
                truths_in_bin = truths[(sigmas >= left[i] ) & (sigmas < left[i+1] )]
                # print(sum((sigmas >= left[i]) & (sigmas < left[i+1])))
                if i == self.num_bins - 1:
                    sigmas_in_bin = sigmas[(sigmas >= left[i]) & (sigmas <= left[i+1] )]
                    truths_in_bin = truths[(sigmas >= left[i] ) & (sigmas <= left[i+1] )]
                    # print(sum((sigmas >= left[i] ) & (sigmas <= left[i+1] )))
                print(((sigmas >= left[i]) & (sigmas < left[i+1])).sum())
                mvar = torch.mean(sigmas_in_bin).sqrt()
                rmse = torch.mean(truths_in_bin).sqrt()
                self.intervals_mvar[i]= mvar
                self.intervals_rmse[i] = rmse

            abs_val = (mvar - rmse).abs() / mvar
        # print(self.intervals_mvar)
        # print(self.intervals_rmse)
        # reliability_diagram(self.intervals_mvar, self.intervals_rmse, int(left[0] ** 0.5) , int(left[-1] ** 0.5) )
        # print(abs_val.mean())
        return abs_val.mean()

    def draw(self, sigmas, truths) -> Any:

        num_samples = sigmas.shape[0]
        # average = (sigmas[:, None] + sigmas[None, :]) / 200
        # dist_gt = ((y[:, None, :] - y[None, :, :]) ** 2).sum(dim=-1).sqrt()
        # dist_pred = ((py[:, None, :] - py[None, :, :]) ** 2).sum(dim=-1).sqrt()

        # dim_ = average.shape[0]
        # average = average.flatten()[1:].view(dim_-1, dim_+1)[:,:-1].reshape(dim_, dim_-1).flatten()
        # dist_gt = dist_gt.flatten()[1:].view(dim_-1, dim_+1)[:,:-1].reshape(dim_, dim_-1).flatten()
        # dist_pred = dist_pred.flatten()[1:].view(dim_-1, dim_+1)[:,:-1].reshape(dim_, dim_-1).flatten()
        # truths = dist_gt - dist_pred
        with torch.no_grad():
            ids = torch.argsort(sigmas, dim=0)
            sigmas = torch.pow(sigmas[ids], 2)
            truths = torch.pow(truths[ids], 2)
            num = num_samples / self.num_bins

            begin = sigmas[0]
            end = sigmas[-1]
            interval = (end - begin) / self.num_bins
            left = [begin + i * interval for i in range(self.num_bins)]
            left.append(begin + self.num_bins * interval + 1)

            for i in range(self.num_bins):
            
                sigmas_in_bin = sigmas[(sigmas >= left[i]) & (sigmas < left[i+1] )]
                truths_in_bin = truths[(sigmas >= left[i] ) & (sigmas < left[i+1] )]
                # print(sum((sigmas >= left[i]) & (sigmas < left[i+1])))
                if i == self.num_bins - 1:
                    sigmas_in_bin = sigmas[(sigmas >= left[i]) & (sigmas <= left[i+1] )]
                    truths_in_bin = truths[(sigmas >= left[i] ) & (sigmas <= left[i+1] )]
                    # print(sum((sigmas >= left[i] ) & (sigmas <= left[i+1] )))

                mvar = torch.mean(sigmas_in_bin).sqrt()
                rmse = torch.mean(truths_in_bin).sqrt()
                self.intervals_mvar[i]= mvar
                self.intervals_rmse[i] = rmse

            abs_val = (mvar - rmse).abs() / mvar
        # print(self.intervals_mvar)
        # print(self.intervals_rmse)
        # reliability_diagram(self.intervals_mvar, self.intervals_rmse, int(left[0] ** 0.5) , int(left[-1] ** 0.5) )
        # print(abs_val.mean())
        return abs_val.mean()


class STDScaling(nn.Module):
    def __init__(self, dev, metric, method, select='ae') -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.)
        self.dev = dev
        self.metric = metric
        self.select = select
        self.optimizer = torch.optim.LBFGS([self.temperature], lr=0.1, max_iter=100)
        
        self.method = method
    
    def loss_function3(self, gt_list, pred_list, y_list, py_list):
        error = (self.temperature * pred_list) - gt_list
        return (error ** 2).mean() 
    
    def loss_function_lddt(self, plddt_fs, dist_list, dist_list_pred):
        plddt_fs = torch.clamp(plddt_fs * self.temperature, 0, 100)
        average_list = (plddt_fs) / 100
        first = -0.5 * torch.log(-torch.log(1 - average_list ** 2 + 1e-12))
        second = -torch.log(1 - average_list ** 2 + 1e-12) / 4 * ((dist_list - dist_list_pred) ** 2)
        return (first + second).mean()

    def set_temeprature(self, valid_loader):

        if self.select == 'lddt':
            print(f'calibrate {self.select}...')
            self.to(self.dev)
            gt_list, pred_list, dist_list, dist_list_pred, plddt_fs = self.load_data(valid_loader)
            def eval():
                self.optimizer.zero_grad()
                if self.method == 'nll':
                    loss = self.loss_function_lddt(plddt_fs, dist_list, dist_list_pred)
                elif self.method == 'reg':
                    loss = self.loss_function3(gt_list, pred_list, None, None)
                loss.backward()
                print(loss)
                return loss
            # self.temperature = nn.Parameter(torch.ones(1) * 0.5).to(self.dev)
            self.optimizer.step(eval)
            print(f'==> temperature value is {self.temperature.data}')
            self.metric.set_temperature(self.temperature.data.cpu())

    def load_data(self, valid_loader):

        if self.select == 'lddt':
            # average_list = []
            dist_list = []
            dist_list_pred = []

            gt_list = []
            pred_list = []
            
            plddt_fs = []
            # plddt_ss = []
            for y, py, lddt, plddt in valid_loader:
                lddt, plddt = lddt.flatten().to(self.dev), plddt.flatten().to(self.dev)
                y, py = y.reshape(-1, 3).to(self.dev), py.reshape(-1, 3).to(self.dev)


                dist_gt = ((y[:, None, :] - y[None, :, :]) ** 2).sum(dim=-1).sqrt().flatten()
                dist_pred = ((py[:, None, :] - py[None, :, :]) ** 2).sum(dim=-1).sqrt().flatten()
                # dim_ = dist_gt.shape[0]
                # dist_gt = dist_gt.flatten()[1:].view(dim_-1, dim_+1)[:,:-1].reshape(dim_, dim_-1)
                # dist_pred = dist_pred.flatten()[1:].view(dim_-1, dim_+1)[:,:-1].reshape(dim_, dim_-1)

                plddt_first = plddt[:, None].repeat(1, plddt.shape[0]).flatten()
                # plddt_sec = plddt[None, :].repeat(plddt.shape[0], 1)

                dist_list.append(dist_gt)
                dist_list_pred.append(dist_pred)
                # plddt_list.append(plddt)

                gt_list.append(lddt)
                pred_list.append(plddt)
                # y_list.append(y)
                # py_list.append(py)

                plddt_fs.append(plddt_first)
                # plddt_ss.append(plddt_sec)
                # print(torch.mean((dist_gt - dist_pred) ** 2), keys)


            gt_list = torch.cat(gt_list, dim=0).to(self.dev)
            pred_list = torch.cat(pred_list, dim=0).to(self.dev)
            # average_list = torch.cat(average_list, dim=0)#.to(self.dev)
            dist_list = torch.cat(dist_list, dim=0).to(self.dev)
            dist_list_pred = torch.cat(dist_list_pred, dim=0).to(self.dev)
            # plddt_list = torch.cat(plddt_list, dim=0).to(self.dev)
            plddt_fs = torch.cat(plddt_fs, dim=0).to(self.dev)
            # plddt_ss = torch.cat(plddt_ss, dim=0).to(self.dev)


        else:
            NotImplementedError

        if self.select == 'lddt':
            return gt_list, pred_list, dist_list, dist_list_pred, plddt_fs
        





#         mask_pred = (dist_pred > 15)
# plddt_first = plddt_first[mask_pred]
# plddt_sec = plddt_sec[mask_pred]
# dist_gt = dist_gt[mask_pred]
# dist_pred = dist_pred[mask_pred]
# # average = average[mask_pred]