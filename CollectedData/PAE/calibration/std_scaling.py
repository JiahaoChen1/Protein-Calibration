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
    
    def loss_function_ae(self, gt_list, pred_list, y_list, py_list):
        first = torch.log(self.temperature * 2.5066282746310002 * pred_list + 1e-7)
        substract = ((y_list - py_list) ** 2).sum(dim=-1)
        second = 0.5 * substract / (((self.temperature * pred_list) ** 2) + 1e-7)
        return (first + second).mean()

    def set_temeprature(self, valid_loader):
        if self.select == 'ae':
            print(f'calibrate {self.select}...')
            self.to(self.dev)
            gt_list, pred_list, y_list, py_list = self.load_data(valid_loader)
            # if self.select == 'lddt':
            #     gt_list, pred_list = gt_list / 100, pred_list / 100
            T = gt_list.shape[0]
            
            def eval():
                self.optimizer.zero_grad()
                if self.method == 'nll':
                    loss = self.loss_function_ae(gt_list, pred_list, y_list, py_list)
                elif self.method == 'reg':
                    loss = self.loss_function3(gt_list, pred_list, y_list, py_list)
                loss.backward()
                print(loss)
                return loss
            self.optimizer.step(eval)
            print(f'==> temperature value is {self.temperature.data}')
            self.metric.set_temperature(self.temperature.data.cpu())

    def load_data(self, valid_loader):
        if self.select == 'ae':
            gt_list = []
            pred_list = []
            y_list = []
            py_list = []
            for y, py, ae, pae in valid_loader:
                ae, pae = ae.flatten(), pae.flatten()
                y, py = y.reshape(-1, 3), py.reshape(-1, 3)
                gt_list.append(ae)
                pred_list.append(pae)
                y_list.append(y)
                py_list.append(py)

            gt_list = torch.cat(gt_list, dim=0).to(self.dev)
            pred_list = torch.cat(pred_list, dim=0).to(self.dev)
            y_list = torch.cat(y_list, dim=0).to(self.dev)
            py_list = torch.cat(py_list, dim=0).to(self.dev)
            print(f'load data to cuda successfully, training samples {pred_list.shape[0]}...')

        else:
            NotImplementedError

        
        if self.select == 'ae':
            return gt_list, pred_list, y_list, py_list

        





#         mask_pred = (dist_pred > 15)
# plddt_first = plddt_first[mask_pred]
# plddt_sec = plddt_sec[mask_pred]
# dist_gt = dist_gt[mask_pred]
# dist_pred = dist_pred[mask_pred]
# # average = average[mask_pred]