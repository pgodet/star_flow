from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf
import scipy.misc
import os
import numpy as np


def _elementwise_epe(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.norm(residual, p=2, dim=1, keepdim=True)

def _elementwise_robust_epe_char(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.pow(torch.norm(residual, p=2, dim=1, keepdim=True) + 0.01, 0.4)

def _downsample2d_as(inputs, target_as):
    _, _, h, w = target_as.size()
    return tf.adaptive_avg_pool2d(inputs, [h, w])

def _upsample2d_as(inputs, target_as, mode="bilinear"):
    _, _, h, w = target_as.size()
    return tf.interpolate(inputs, [h, w], mode=mode, align_corners=True)

def f1_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, 1)

def fbeta_score(y_true, y_pred, beta, eps=1e-8):
    beta2 = beta ** 2

    y_pred = y_pred.float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=2).sum(dim=2)
    precision = true_positive / (y_pred.sum(dim=2).sum(dim=2) + eps)
    recall = true_positive / (y_true.sum(dim=2).sum(dim=2) + eps)

    return torch.mean(precision * recall / (precision * beta2 + recall + eps) * (1 + beta2))

def f1_score_bal_loss(y_pred, y_true):
    eps = 1e-8

    tp = -(y_true * torch.log(y_pred + eps)).sum(dim=2).sum(dim=2).sum(dim=1)
    fn = -((1 - y_true) * torch.log((1 - y_pred) + eps)).sum(dim=2).sum(dim=2).sum(dim=1)

    denom_tp = y_true.sum(dim=2).sum(dim=2).sum(dim=1) + y_pred.sum(dim=2).sum(dim=2).sum(dim=1) + eps
    denom_fn = (1 - y_true).sum(dim=2).sum(dim=2).sum(dim=1) + (1 - y_pred).sum(dim=2).sum(dim=2).sum(dim=1) + eps

    return ((tp / denom_tp).sum() + (fn / denom_fn).sum()) * y_pred.size(2) * y_pred.size(3) * 0.5


class MultiScaleEPE_PWC(nn.Module):
    def __init__(self, args):

        super(MultiScaleEPE_PWC, self).__init__()
        self._args = args
        self._batch_size = args.batch_size
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005]

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            outputs = output_dict['flow']

            # div_flow trick
            target = self._args.model_div_flow * target_dict["target1"]

            total_loss = 0
            for ii, output_ii in enumerate(outputs):
                loss_ii = _elementwise_epe(output_ii, _downsample2d_as(target, output_ii)).sum()
                total_loss = total_loss + self._weights[ii] * loss_ii
            loss_dict["total_loss"] = total_loss / self._batch_size

        else:
            epe = _elementwise_epe(output_dict["flow"], target_dict["target1"])
            loss_dict["epe"] = epe.mean()

        return loss_dict


class MultiScaleEPE_PWC_Sintel(nn.Module):
    def __init__(self, args):

        super(MultiScaleEPE_PWC_Sintel, self).__init__()
        self._args = args
        self._batch_size = args.batch_size
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005]

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            outputs = output_dict['flow']

            # div_flow trick
            target = self._args.model_div_flow * target_dict["target1"]

            total_loss = 0
            for ii, output_ii in enumerate(outputs):
                loss_ii = _elementwise_robust_epe_char(output_ii, _downsample2d_as(target, output_ii)).sum()
                total_loss = total_loss + self._weights[ii] * loss_ii
            loss_dict["total_loss"] = total_loss / self._batch_size

        else:
            epe = _elementwise_epe(output_dict["flow"], target_dict["target1"])
            loss_dict["epe"] = epe.mean()

        return loss_dict

class MultiScaleEPE_PWC_Kitti(nn.Module):
    def __init__(self, args):

        super(MultiScaleEPE_PWC_Kitti, self).__init__()
        self._args = args
        self._batch_size = args.batch_size
        self._weights = [0.001, 0.001, 0.001, 0.002, 0.004]

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        valid_mask = target_dict["input_valid"]
        b, _, h, w = target_dict["target1"].size()

        if self.training:
            outputs = output_dict['flow']

            # div_flow trick
            target = self._args.model_div_flow * target_dict["target1"]

            total_loss = 0
            for ii, output_ii in enumerate(outputs):
                loss_ii = 0
                if output_ii.ndimension() > 4:
                    valid_epe = _elementwise_robust_epe_char(_upsample2d_as(output_ii[-1], target), target) * valid_mask
                else:
                    valid_epe = _elementwise_robust_epe_char(_upsample2d_as(output_ii, target), target) * valid_mask

                for bb in range(0, b):
                    valid_epe[bb, ...][valid_mask[bb, ...] == 0] = valid_epe[bb, ...][valid_mask[bb, ...] == 0].detach()
                    norm_const = h * w / (valid_mask[bb, ...].sum())
                    loss_ii = loss_ii + valid_epe[bb, ...][valid_mask[bb, ...] != 0].sum() * norm_const

                total_loss = total_loss + self._weights[ii] * loss_ii

            loss_dict["total_loss"] = total_loss / self._batch_size

        else:
            flow_gt_mag = torch.norm(target_dict["target1"], p=2, dim=1, keepdim=True) + 1e-8
            if output_dict["flow"].ndimension() > 4:
                flow_epe = _elementwise_epe(output_dict["flow"][-1], target_dict["target1"]) * valid_mask
            else:
                flow_epe = _elementwise_epe(output_dict["flow"], target_dict["target1"]) * valid_mask

            epe_per_image = (flow_epe.view(b, -1).sum(1)) / (valid_mask.view(b, -1).sum(1))
            loss_dict["epe"] = epe_per_image.mean()

            outlier_epe = (flow_epe > 3).float() * ((flow_epe / flow_gt_mag) > 0.05).float() * valid_mask
            outlier_per_image = (outlier_epe.view(b, -1).sum(1)) / (valid_mask.view(b, -1).sum(1))
            loss_dict["outlier"] = outlier_per_image.mean()

        return loss_dict

class MultiScaleEPE_PWC_video(nn.Module):
    def __init__(self, args):

        super(MultiScaleEPE_PWC_video, self).__init__()
        self._args = args
        self._batch_size = args.batch_size
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005]

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            outputs = output_dict['flow']

            # div_flow trick
            targets = self._args.model_div_flow * torch.stack(target_dict["target_flows"], 0)

            total_loss = 0
            for ii, output_ii in enumerate(outputs):
                nframes = output_ii.size(0)
                loss_ii = 0
                for n in range(nframes):
                    loss_ii_n = _elementwise_epe(output_ii[n], _downsample2d_as(targets[n], output_ii[n])).sum()
                    loss_ii = loss_ii + loss_ii_n
                loss_ii = loss_ii / nframes
                total_loss = total_loss + self._weights[ii] * loss_ii
            loss_dict["total_loss"] = total_loss / self._batch_size

        else:
            # compute error for last time step of subsequence:
            loss_dict["epe"] = _elementwise_epe(output_dict["flow"][-1], target_dict["target_flows"][-1]).mean()

        return loss_dict


class MultiScaleEPE_PWC_Bi(nn.Module):
    def __init__(self, args):

        super(MultiScaleEPE_PWC_Bi, self).__init__()
        self._args = args
        self._batch_size = args.batch_size
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005]

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            outputs = output_dict['flow']

            # div_flow trick
            target_f = self._args.model_div_flow * target_dict["target1"]
            target_b = self._args.model_div_flow * target_dict["target2"]

            total_loss = 0
            for i, output_i in enumerate(outputs):
                epe_i_f = _elementwise_epe(output_i[0], _downsample2d_as(target_f, output_i[0]))
                epe_i_b = _elementwise_epe(output_i[1], _downsample2d_as(target_b, output_i[1]))
                total_loss = total_loss + self._weights[i] * (epe_i_f.sum() + epe_i_b.sum())
            loss_dict["total_loss"] = total_loss / (2 * self._batch_size)
        else:
            epe = _elementwise_epe(output_dict["flow"], target_dict["target1"])
            loss_dict["epe"] = epe.mean()

        return loss_dict

class MultiScaleEPE_PWC_Occ(nn.Module):
    def __init__(self, args):

        super(MultiScaleEPE_PWC_Occ, self).__init__()
        self._args = args
        self._batch_size = args.batch_size
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005]

        self.occ_activ = nn.Sigmoid()
        self.f1_score_bal_loss = f1_score_bal_loss

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            output_flo = output_dict['flow']
            output_occ = output_dict['occ']

            # div_flow trick
            target_flo = self._args.model_div_flow * target_dict["target1"]
            target_occ = target_dict["target_occ1"]

            flow_loss = 0
            occ_loss = 0

            for i, output_i in enumerate(output_flo):
                flow_loss = flow_loss + self._weights[i] * _elementwise_epe(output_i, _downsample2d_as(target_flo, output_i)).sum()

            for i, output_i in enumerate(output_occ):
                output_occ = self.occ_activ(output_i)
                occ_loss = occ_loss + self._weights[i] * self.f1_score_bal_loss(output_occ, _downsample2d_as(target_occ, output_occ))

            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if (f_loss.data > o_loss.data).numpy:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1

            loss_dict["flow_loss"] = flow_loss / self._batch_size
            loss_dict["occ_loss"] = occ_loss / self._batch_size
            loss_dict["total_loss"] = (flow_loss * f_l_w + occ_loss * o_l_w) / self._batch_size

        else:
            loss_dict["epe"] = _elementwise_epe(output_dict["flow"], target_dict["target1"]).mean()
            loss_dict["F1"] = f1_score(target_dict["target_occ1"], torch.round(self.occ_activ(output_dict["occ"])))

        return loss_dict

class MultiScaleEPE_PWC_Occ_video(nn.Module):
    def __init__(self, args):

        super(MultiScaleEPE_PWC_Occ_video, self).__init__()
        self._args = args
        self._batch_size = args.batch_size
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005]

        self.occ_activ = nn.Sigmoid()
        self.f1_score_bal_loss = f1_score_bal_loss

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            outputs_flo = output_dict['flow']
            outputs_occ = output_dict['occ']

            # div_flow trick
            targets_flo = self._args.model_div_flow * torch.stack(target_dict["target_flows"], 0)
            targets_occ = torch.stack(target_dict["target_occs"], 0)

            flow_loss = 0
            occ_loss = 0

            for ii, output_ii in enumerate(outputs_flo):
                nframes = output_ii.size(0)
                loss_ii = 0
                for n in range(nframes):
                    loss_ii_n = _elementwise_epe(output_ii[n], _downsample2d_as(targets_flo[n], output_ii[n])).sum()
                    loss_ii = loss_ii + loss_ii_n
                loss_ii = loss_ii / nframes
                flow_loss = flow_loss + self._weights[ii] * loss_ii

            for ii, output_ii in enumerate(outputs_occ):
                nframes = output_ii.size(0)
                loss_ii = 0
                for n in range(nframes):
                    output_occ_n = self.occ_activ(output_ii[n])
                    loss_ii_n = self.f1_score_bal_loss(output_occ_n, _downsample2d_as(targets_occ[n], output_occ_n))
                    loss_ii = loss_ii + loss_ii_n
                loss_ii = loss_ii / nframes
                occ_loss = occ_loss + self._weights[ii] * loss_ii

            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if (f_loss.data > o_loss.data).numpy:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1

            loss_dict["flow_loss"] = flow_loss / self._batch_size
            loss_dict["occ_loss"] = occ_loss / self._batch_size
            loss_dict["total_loss"] = (flow_loss * f_l_w + occ_loss * o_l_w) / self._batch_size

        else:
            # compute error for last time step of subsequence:
            loss_dict["epe"] = _elementwise_epe(output_dict["flow"][-1], target_dict["target_flows"][-1]).mean()
            loss_dict["F1"] = f1_score(target_dict["target_occs"][-1], torch.round(self.occ_activ(output_dict["occ"][-1])))

        return loss_dict

class MultiScaleEPE_PWC_Occ_video_Sintel(nn.Module):
    def __init__(self, args):

        super(MultiScaleEPE_PWC_Occ_video_Sintel, self).__init__()
        self._args = args
        self._batch_size = args.batch_size
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005]

        self.occ_activ = nn.Sigmoid()
        self.occ_loss_bce = nn.BCELoss(reduction='sum')

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            outputs_flo = output_dict['flow']
            outputs_occ = output_dict['occ']

            # div_flow trick
            targets_flo = self._args.model_div_flow * torch.stack(target_dict["target_flows"], 0)
            targets_occ = torch.stack(target_dict["target_occs"], 0)

            flow_loss = 0
            occ_loss = 0

            for ii, output_ii in enumerate(outputs_flo):
                nframes = output_ii.size(0)
                loss_ii = 0
                for n in range(nframes):
                    loss_ii_n = _elementwise_robust_epe_char(output_ii[n], _downsample2d_as(targets_flo[n], output_ii[n])).sum()
                    loss_ii = loss_ii + loss_ii_n
                loss_ii = loss_ii / nframes
                flow_loss = flow_loss + self._weights[ii] * loss_ii

            for ii, output_ii in enumerate(outputs_occ):
                nframes = output_ii.size(0)
                loss_ii = 0
                for n in range(nframes):
                    output_occ_n = self.occ_activ(output_ii[n])
                    loss_ii_n = self.occ_loss_bce(output_occ_n, _downsample2d_as(targets_occ[n], output_occ_n))
                    loss_ii = loss_ii + loss_ii_n
                loss_ii = loss_ii / nframes
                occ_loss = occ_loss + self._weights[ii] * loss_ii

            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if (f_loss.data > o_loss.data).numpy:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1

            loss_dict["flow_loss"] = flow_loss / self._batch_size
            loss_dict["occ_loss"] = occ_loss / self._batch_size
            loss_dict["total_loss"] = (flow_loss * f_l_w + occ_loss * o_l_w) / self._batch_size

        else:
            # compute error for last time step of subsequence:
            loss_dict["epe"] = _elementwise_epe(output_dict["flow"][-1], target_dict["target_flows"][-1]).mean()
            loss_dict["F1"] = f1_score(target_dict["target_occs"][-1], torch.round(self.occ_activ(output_dict["occ"][-1])))

        return loss_dict


# ---

class MultiScaleEPE_PWC_Bi_Occ_upsample(nn.Module):
    def __init__(self, args):

        super(MultiScaleEPE_PWC_Bi_Occ_upsample, self).__init__()
        self._args = args
        self._batch_size = args.batch_size
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005, 0.00125, 0.0003125]

        self.occ_activ = nn.Sigmoid()
        self.f1_score_bal_loss = f1_score_bal_loss

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            output_flo = output_dict['flow']
            output_occ = output_dict['occ']

            # div_flow trick
            target_flo_f = self._args.model_div_flow * target_dict["target1"]
            target_flo_b = self._args.model_div_flow * target_dict["target2"]
            target_occ_f = target_dict["target_occ1"]
            target_occ_b = target_dict["target_occ2"]

            # bchw
            flow_loss = 0
            occ_loss = 0

            for ii, output_ii in enumerate(output_flo):
                loss_ii = 0
                for jj in range(0, len(output_ii) // 2):
                    loss_ii = loss_ii + _elementwise_epe(output_ii[2 * jj], _downsample2d_as(target_flo_f, output_ii[2 * jj])).sum()
                    loss_ii = loss_ii + _elementwise_epe(output_ii[2 * jj + 1], _downsample2d_as(target_flo_b, output_ii[2 * jj + 1])).sum()
                flow_loss = flow_loss + self._weights[ii] * loss_ii / len(output_ii)

            for ii, output_ii in enumerate(output_occ):
                loss_ii = 0
                for jj in range(0, len(output_ii) // 2):
                    output_occ_f = self.occ_activ(output_ii[2 * jj])
                    output_occ_b = self.occ_activ(output_ii[2 * jj + 1])
                    loss_ii = loss_ii + self.f1_score_bal_loss(output_occ_f, _downsample2d_as(target_occ_f, output_occ_f))
                    loss_ii = loss_ii + self.f1_score_bal_loss(output_occ_b, _downsample2d_as(target_occ_b, output_occ_b))
                occ_loss = occ_loss + self._weights[ii] * loss_ii / len(output_ii)

            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if (f_loss.data > o_loss.data).numpy:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1

            loss_dict["flow_loss"] = flow_loss / self._batch_size
            loss_dict["occ_loss"] = occ_loss / self._batch_size
            loss_dict["total_loss"] = (flow_loss * f_l_w + occ_loss * o_l_w) / self._batch_size

        else:
            loss_dict["epe"] = _elementwise_epe(output_dict["flow"], target_dict["target1"]).mean()
            loss_dict["F1"] = f1_score(target_dict["target_occ1"], torch.round(self.occ_activ(output_dict["occ"])))

        return loss_dict


class MultiScaleEPE_PWC_Occ_upsample(nn.Module):
    def __init__(self, args):

        super(MultiScaleEPE_PWC_Occ_upsample, self).__init__()
        self._args = args
        self._batch_size = args.batch_size
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005, 0.00125, 0.0003125]

        self.occ_activ = nn.Sigmoid()
        self.f1_score_bal_loss = f1_score_bal_loss

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            output_flo = output_dict['flow']
            output_occ = output_dict['occ']
            output_flo_coarse = output_dict['flow_coarse']
            output_occ_coarse = output_dict['occ_coarse']

            # div_flow trick
            target_flo = self._args.model_div_flow * target_dict["target1"]
            target_occ = target_dict["target_occ1"]

            flow_loss = 0
            occ_loss = 0

            for i, output_i in enumerate(output_flo):
                if i < len(output_flo_coarse):
                    temp1 = self._weights[i] * _elementwise_epe(output_flo_coarse[i], _downsample2d_as(target_flo, output_flo_coarse[i])).sum()
                    temp2 = self._weights[i] * _elementwise_epe(output_i, _downsample2d_as(target_flo, output_i)).sum()
                    flow_loss = flow_loss + (temp1 + temp2) / 2
                else:
                    flow_loss = flow_loss + self._weights[i] * _elementwise_epe(output_i, _downsample2d_as(target_flo, output_i)).sum()

            for i, output_i in enumerate(output_occ):
                if i < len(output_occ_coarse):
                    output_occ = self.occ_activ(output_occ_coarse[i])
                    temp1 = self._weights[i] * self.f1_score_bal_loss(output_occ, _downsample2d_as(target_occ, output_occ))
                    output_occ = self.occ_activ(output_i)
                    temp2 = self._weights[i] * self.f1_score_bal_loss(output_occ, _downsample2d_as(target_occ, output_occ))
                    occ_loss = occ_loss + (temp1 + temp2) / 2
                else:
                    output_occ = self.occ_activ(output_i)
                    occ_loss = occ_loss + self._weights[i] * self.f1_score_bal_loss(output_occ, _downsample2d_as(target_occ, output_occ))

            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if (f_loss.data > o_loss.data).numpy:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1

            loss_dict["flow_loss"] = flow_loss / self._batch_size
            loss_dict["occ_loss"] = occ_loss / self._batch_size
            loss_dict["total_loss"] = (flow_loss * f_l_w + occ_loss * o_l_w) / self._batch_size

        else:
            loss_dict["epe"] = _elementwise_epe(output_dict["flow"], target_dict["target1"]).mean()
            loss_dict["F1"] = f1_score(target_dict["target_occ1"], torch.round(self.occ_activ(output_dict["occ"])))

        return loss_dict


class MultiScaleEPE_PWC_video_upsample(nn.Module):
    def __init__(self, args):

        super(MultiScaleEPE_PWC_video_upsample, self).__init__()
        self._args = args
        self._batch_size = args.batch_size
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005, 0.00125, 0.0003125]

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            outputs_flo = output_dict['flow']
            outputs_flo_coarse = output_dict['flow_coarse']

            # div_flow trick
            targets_flo = self._args.model_div_flow * torch.stack(target_dict["target_flows"], 0)

            flow_loss = 0

            for ii, output_ii in enumerate(outputs_flo):
                nframes = output_ii.size(0)
                loss_ii = 0
                for n in range(nframes):
                    if ii < len(outputs_flo_coarse):
                        loss_ii_n = 0
                        loss_ii_n = loss_ii_n + _elementwise_epe(outputs_flo_coarse[ii][n], _downsample2d_as(targets_flo[n], outputs_flo_coarse[ii][n])).sum()
                        loss_ii_n = loss_ii_n + _elementwise_epe(output_ii[n], _downsample2d_as(targets_flo[n], output_ii[n])).sum()
                        loss_ii = loss_ii + loss_ii_n / 2
                    else:
                        loss_ii_n = _elementwise_epe(output_ii[n], _downsample2d_as(targets_flo[n], output_ii[n])).sum()
                        loss_ii = loss_ii + loss_ii_n
                loss_ii = loss_ii / nframes
                flow_loss = flow_loss + self._weights[ii] * loss_ii

            #loss_dict["flow_loss"] =
            loss_dict["total_loss"] = flow_loss / self._batch_size

        else:
            # compute error for last time step of subsequence:
            loss_dict["epe"] = _elementwise_epe(output_dict["flow"][-1], target_dict["target_flows"][-1]).mean()
            #loss_dict["F1"] = f1_score(target_dict["target_occs"][-1], torch.round(self.occ_activ(output_dict["occ"][-1])))
            # epe = 0
            # for n, target_n in enumerate(target_dict["target_flows"]):
            #     epe += _elementwise_epe(output_dict["flow"][n], target_n).mean()
            # loss_dict["epe"] = epe / (n + 1)
            # F1 = 0
            # for n, target_n in enumerate(target_dict["target_occs"]):
            #     F1 += f1_score(target_n, torch.round(self.occ_activ(output_dict["occ"][n])))
            # loss_dict["F1"] = F1 / (n + 1)

        return loss_dict

class MultiScaleEPE_PWC_Occ_video_upsample(nn.Module):
    def __init__(self, args):

        super(MultiScaleEPE_PWC_Occ_video_upsample, self).__init__()
        self._args = args
        self._batch_size = args.batch_size
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005, 0.00125, 0.0003125]

        self.occ_activ = nn.Sigmoid()
        self.f1_score_bal_loss = f1_score_bal_loss

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            outputs_flo = output_dict['flow']
            outputs_occ = output_dict['occ']
            outputs_flo_coarse = output_dict['flow_coarse']
            outputs_occ_coarse = output_dict['occ_coarse']

            # div_flow trick
            targets_flo = self._args.model_div_flow * torch.stack(target_dict["target_flows"], 0)
            targets_occ = torch.stack(target_dict["target_occs"], 0)

            flow_loss = 0
            occ_loss = 0

            for ii, output_ii in enumerate(outputs_flo):
                nframes = output_ii.size(0)
                loss_ii = 0
                for n in range(nframes):
                    if ii < len(outputs_flo_coarse):
                        loss_ii_n = 0
                        loss_ii_n = loss_ii_n + _elementwise_epe(outputs_flo_coarse[ii][n], _downsample2d_as(targets_flo[n], outputs_flo_coarse[ii][n])).sum()
                        loss_ii_n = loss_ii_n + _elementwise_epe(output_ii[n], _downsample2d_as(targets_flo[n], output_ii[n])).sum()
                        loss_ii = loss_ii + loss_ii_n / 2
                    else:
                        loss_ii_n = _elementwise_epe(output_ii[n], _downsample2d_as(targets_flo[n], output_ii[n])).sum()
                        loss_ii = loss_ii + loss_ii_n
                loss_ii = loss_ii / nframes
                flow_loss = flow_loss + self._weights[ii] * loss_ii

            for ii, output_ii in enumerate(outputs_occ):
                nframes = output_ii.size(0)
                loss_ii = 0
                for n in range(nframes):
                    if ii < len(outputs_occ_coarse):
                        loss_ii_n = 0
                        output_occ_n = self.occ_activ(outputs_occ_coarse[ii][n])
                        loss_ii_n = loss_ii_n + self.f1_score_bal_loss(output_occ_n, _downsample2d_as(targets_occ[n], output_occ_n))
                        output_occ_n = self.occ_activ(output_ii[n])
                        loss_ii_n = loss_ii_n + self.f1_score_bal_loss(output_occ_n, _downsample2d_as(targets_occ[n], output_occ_n))
                        loss_ii = loss_ii + loss_ii_n / 2
                    else:
                        output_occ_n = self.occ_activ(output_ii[n])
                        loss_ii_n = self.f1_score_bal_loss(output_occ_n, _downsample2d_as(targets_occ[n], output_occ_n))
                        loss_ii = loss_ii + loss_ii_n
                loss_ii = loss_ii / nframes
                occ_loss = occ_loss + self._weights[ii] * loss_ii

            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if (f_loss.data > o_loss.data).numpy:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1

            loss_dict["flow_loss"] = flow_loss / self._batch_size
            loss_dict["occ_loss"] = occ_loss / self._batch_size
            loss_dict["total_loss"] = (flow_loss * f_l_w + occ_loss * o_l_w) / self._batch_size

        else:
            # compute error for last time step of subsequence:
            loss_dict["epe"] = _elementwise_epe(output_dict["flow"][-1], target_dict["target_flows"][-1]).mean()
            loss_dict["F1"] = f1_score(target_dict["target_occs"][-1], torch.round(self.occ_activ(output_dict["occ"][-1])))
            # epe = 0
            # for n, target_n in enumerate(target_dict["target_flows"]):
            #     epe += _elementwise_epe(output_dict["flow"][n], target_n).mean()
            # loss_dict["epe"] = epe / (n + 1)
            # F1 = 0
            # for n, target_n in enumerate(target_dict["target_occs"]):
            #     F1 += f1_score(target_n, torch.round(self.occ_activ(output_dict["occ"][n])))
            # loss_dict["F1"] = F1 / (n + 1)

        return loss_dict

class MultiScaleEPE_PWC_Occ_video_upsample_Sintel(nn.Module):
    def __init__(self, args):

        super(MultiScaleEPE_PWC_Occ_video_upsample_Sintel, self).__init__()
        self._args = args
        self._batch_size = args.batch_size
        self._weights = [0.32, 0.08, 0.02, 0.01, 0.005, 0.00125, 0.0003125]

        self.occ_activ = nn.Sigmoid()
        self.occ_loss_bce = nn.BCELoss(reduction='sum')

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            outputs_flo = output_dict['flow']
            outputs_occ = output_dict['occ']
            outputs_flo_coarse = output_dict['flow_coarse']
            outputs_occ_coarse = output_dict['occ_coarse']

            # div_flow trick
            targets_flo = self._args.model_div_flow * torch.stack(target_dict["target_flows"], 0)
            targets_occ = torch.stack(target_dict["target_occs"], 0)

            flow_loss = 0
            occ_loss = 0

            for ii, output_ii in enumerate(outputs_flo):
                nframes = output_ii.size(0)
                loss_ii = 0
                for n in range(nframes):
                    if ii < len(outputs_flo_coarse):
                        loss_ii_n = 0
                        loss_ii_n = loss_ii_n + _elementwise_epe(outputs_flo_coarse[ii][n], _downsample2d_as(targets_flo[n], outputs_flo_coarse[ii][n])).sum()
                        loss_ii_n = loss_ii_n + _elementwise_epe(output_ii[n], _downsample2d_as(targets_flo[n], output_ii[n])).sum()
                        loss_ii = loss_ii + loss_ii_n / 2
                    else:
                        loss_ii_n = _elementwise_epe(output_ii[n], _downsample2d_as(targets_flo[n], output_ii[n])).sum()
                        loss_ii = loss_ii + loss_ii_n
                loss_ii = loss_ii / nframes
                flow_loss = flow_loss + self._weights[ii] * loss_ii

            for ii, output_ii in enumerate(outputs_occ):
                nframes = output_ii.size(0)
                loss_ii = 0
                for n in range(nframes):
                    if ii < len(outputs_occ_coarse):
                        loss_ii_n = 0
                        output_occ_n = self.occ_activ(outputs_occ_coarse[ii][n])
                        loss_ii_n = loss_ii_n + self.occ_loss_bce(output_occ_n, _downsample2d_as(targets_occ[n], output_occ_n))
                        output_occ_n = self.occ_activ(output_ii[n])
                        loss_ii_n = loss_ii_n + self.occ_loss_bce(output_occ_n, _downsample2d_as(targets_occ[n], output_occ_n))
                        loss_ii = loss_ii + loss_ii_n / 2
                    else:
                        output_occ_n = self.occ_activ(output_ii[n])
                        loss_ii_n = self.occ_loss_bce(output_occ_n, _downsample2d_as(targets_occ[n], output_occ_n))
                        loss_ii = loss_ii + loss_ii_n
                loss_ii = loss_ii / nframes
                occ_loss = occ_loss + self._weights[ii] * loss_ii

            f_loss = flow_loss.detach()
            o_loss = occ_loss.detach()
            if (f_loss.data > o_loss.data).numpy:
                f_l_w = 1
                o_l_w = f_loss / o_loss
            else:
                f_l_w = o_loss / f_loss
                o_l_w = 1

            loss_dict["flow_loss"] = flow_loss / self._batch_size
            loss_dict["occ_loss"] = occ_loss / self._batch_size
            loss_dict["total_loss"] = (flow_loss * f_l_w + occ_loss * o_l_w) / self._batch_size

        else:
            # compute error for last time step of subsequence:
            loss_dict["epe"] = _elementwise_epe(output_dict["flow"][-1], target_dict["target_flows"][-1]).mean()
            loss_dict["F1"] = f1_score(target_dict["target_occs"][-1], torch.round(self.occ_activ(output_dict["occ"][-1])))
            # epe = 0
            # for n, target_n in enumerate(target_dict["target_flows"]):
            #     epe += _elementwise_epe(output_dict["flow"][n], target_n).mean()
            # loss_dict["epe"] = epe / (n + 1)
            # F1 = 0
            # for n, target_n in enumerate(target_dict["target_occs"]):
            #     F1 += f1_score(target_n, torch.round(self.occ_activ(output_dict["occ"][n])))
            # loss_dict["F1"] = F1 / (n + 1)

        return loss_dict


class MultiScaleEPE_PWC_Occ_upsample_KITTI(nn.Module):
    """
    Written for StarFlow (with N>2)
    """
    def __init__(self, args):

        super(MultiScaleEPE_PWC_Occ_upsample_KITTI, self).__init__()
        self._args = args
        self._batch_size = args.batch_size
        self._weights = [0.001, 0.001, 0.001, 0.002, 0.004, 0.004, 0.004]

        self.occ_activ = nn.Sigmoid()

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        valid_mask = target_dict["input_valid"]
        b, _, h, w = target_dict["target1"].size()

        if self.training:
            output_flo = output_dict['flow']
            output_occ = output_dict['occ']
            outputs_flo_coarse = output_dict['flow_coarse']
            outputs_occ_coarse = output_dict['occ_coarse']

            # div_flow trick
            target_flo_f = self._args.model_div_flow * target_dict["target1"]

            # bchw
            flow_loss = 0

            for ii, output_ii in enumerate(output_flo):
                loss_ii = 0
                valid_epe = _elementwise_robust_epe_char(_upsample2d_as(output_ii[-1], target_flo_f), target_flo_f) * valid_mask
                if ii < len(outputs_flo_coarse):
                    valid_epe_coarse = _elementwise_robust_epe_char(_upsample2d_as(outputs_flo_coarse[ii][-1], target_flo_f), target_flo_f) * valid_mask

                    for bb in range(0, b):
                        valid_epe[bb, ...][valid_mask[bb, ...] == 0] = valid_epe[bb, ...][valid_mask[bb, ...] == 0].detach()
                        norm_const = h * w / (valid_mask[bb, ...].sum())
                        if ii < len(outputs_flo_coarse):
                            valid_epe_coarse[bb, ...][valid_mask[bb, ...] == 0] = valid_epe_coarse[bb, ...][valid_mask[bb, ...] == 0].detach()
                            loss_ii = loss_ii + valid_epe[bb, ...][valid_mask[bb, ...] != 0].sum() * norm_const / 2
                            loss_ii = loss_ii + valid_epe_coarse[bb, ...][valid_mask[bb, ...] != 0].sum() * norm_const / 2
                        else:
                            loss_ii = loss_ii + valid_epe[bb, ...][valid_mask[bb, ...] != 0].sum() * norm_const

                    #output_ii[2 * jj + 1] = output_ii[2 * jj + 1].detach()
                flow_loss = flow_loss + self._weights[ii] * loss_ii

            for ii, output_ii in enumerate(output_occ):
                nframes = output_ii.size(0)
                for n in range(nframes):
                    output_ii[n] = output_ii[n].detach()

            loss_dict["flow_loss"] = flow_loss / self._batch_size
            loss_dict["total_loss"] = flow_loss / self._batch_size

        else:
            flow_gt_mag = torch.norm(target_dict["target1"], p=2, dim=1, keepdim=True) + 1e-8
            flow_epe = _elementwise_epe(output_dict["flow"][-1], target_dict["target1"]) * valid_mask

            epe_per_image = (flow_epe.view(b, -1).sum(1)) / (valid_mask.view(b, -1).sum(1))
            loss_dict["epe"] = epe_per_image.mean()

            outlier_epe = (flow_epe > 3).float() * ((flow_epe / flow_gt_mag) > 0.05).float() * valid_mask
            outlier_per_image = (outlier_epe.view(b, -1).sum(1)) / (valid_mask.view(b, -1).sum(1))
            loss_dict["outlier"] = outlier_per_image.mean()

        return loss_dict
