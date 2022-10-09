from __future__ import print_function, absolute_import
import torch
import torch.nn.functional as F
from torch import nn, autograd
from torch.autograd import Variable, Function
import numpy as np


torch.autograd.set_detect_anomaly(True)


class ExemplarMemory(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None



class ClusterMemoryBaseline(nn.Module):
    def __init__(self, temp=0.05, momentum=0.2):
        super(ClusterMemoryBaseline, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.temp = temp
        self.momentum = momentum

    def forward(self, features, targets):
        inputs = ExemplarMemory.apply(features, targets, self.cluster_memory, torch.tensor(self.momentum).to(torch.device('cuda')))
        inputs /= self.temp  # similarity score before softmax
        loss = F.cross_entropy(inputs, targets)
        return loss


class CameraAwareMemory(nn.Module):
    def __init__(self, temp=0.05, momentum=0.2, all_img_cams='', has_intra_cam_loss=False, has_cross_cam_loss=False,
                 has_online_proxy_loss=False, bg_knn=50, posK=3, balance_w=0.15):
        super(CameraAwareMemory, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.temp = temp
        self.momentum = momentum
        self.all_img_cams = all_img_cams
        self.unique_cameras = torch.unique(self.all_img_cams)
        self.proxy_cam_dict = {}
        self.all_pseudo_label = ''
        self.all_proxy_label = ''
        self.proxy_label_dict = {}
        self.has_intra_cam_loss = has_intra_cam_loss
        self.has_cross_cam_loss = has_cross_cam_loss
        self.has_online_proxy_loss = has_online_proxy_loss
        self.bg_knn = bg_knn
        self.posK = posK
        self.balance_w = balance_w

    def forward(self, features, targets, cams, epoch):

        pseudo_y = self.all_pseudo_label[targets].to(torch.device('cuda'))  # targets: image index in the train set
        proxy_targets = self.all_proxy_label[targets].to(torch.device('cuda'))
        loss = torch.tensor(0.).to(torch.device('cuda'))

        score = ExemplarMemory.apply(features, proxy_targets, self.global_memory, torch.tensor(self.momentum).to(torch.device('cuda')))
        #score = ExemplarMemory(self.global_memory, alpha=self.momentum)(features, proxy_targets)
        inputs = score / self.temp  # similarity score before softmax

        intra_loss, offline_loss, online_loss = 0, 0, 0
        for cc in torch.unique(cams):
                inds = torch.nonzero(cams == cc).squeeze(-1)
                percam_inputs = inputs[inds]
                percam_y = pseudo_y[inds]
                pc_prx_target = proxy_targets[inds]

                if self.has_intra_cam_loss:
                    intra_loss += self.get_intra_loss_all_cams(percam_inputs, pc_prx_target, cc)

                if self.has_cross_cam_loss:
                    offline_loss += self.get_proxy_associate_loss(percam_inputs, percam_y)
 
                if self.has_online_proxy_loss:
                    temp_score = score[inds].detach().clone()  # for similarity
                    online_loss += self.get_proxy_cam_wise_nn_enhance_loss(temp_score, percam_inputs, pc_prx_target)
   
        loss += (intra_loss + offline_loss + online_loss)
   
        return loss


    def get_intra_loss_all_cams(self, inputs, target_proxy, cams):
        loss = 0
        for i in range(len(inputs)):
            intra_prxs = self.proxy_cam_dict[int(cams[i])]
            sel_input = inputs[i, intra_prxs]
            sel_target = torch.zeros((len(sel_input)), dtype=inputs.dtype).to(torch.device('cuda'))
            sel_target[intra_prxs==target_proxy[i]] = 1.0
            loss += -1.0 * (F.log_softmax(sel_input.unsqueeze(0), dim=1) * sel_target.unsqueeze(0)).sum()
        return loss/len(inputs)


    def get_proxy_associate_loss(self, inputs, targets):
        temp_inputs = inputs.detach().clone()
        loss = 0
        for i in range(len(inputs)):
            pos_ind = self.proxy_label_dict[int(targets[i])]
            temp_inputs[i, pos_ind] = 10000.0  # mask the positives
            sel_ind = torch.sort(temp_inputs[i])[1][-self.bg_knn-len(pos_ind):]
            sel_input = inputs[i, sel_ind]
            sel_target = torch.zeros((len(sel_input)), dtype=sel_input.dtype).to(torch.device('cuda'))
            sel_target[-len(pos_ind):] = 1.0 / len(pos_ind)
            loss += -1.0 * (F.log_softmax(sel_input.unsqueeze(0), dim=1) * sel_target.unsqueeze(0)).sum()
        loss /= len(inputs)
        return loss


    def get_proxy_cam_wise_nn_enhance_loss(self, temp_score, inputs, proxy_targets):

        temp_memory = self.global_memory.detach().clone()  # global_memory is the proxy memory
        soft_target = torch.zeros(self.bg_knn + self.posK, dtype=torch.float).to(torch.device('cuda'))
        soft_target[-self.posK:] = 1.0 / self.posK
        loss = 0

        for i in range(len(inputs)):
            lbl = proxy_targets[i]
            sims = self.balance_w * temp_score[i] + (1 - self.balance_w) * torch.matmul(temp_memory[lbl], temp_memory.t())

            all_cam_tops = []
            for cc in self.unique_cameras:
                proxy_inds = self.proxy_cam_dict[int(cc)]
                maxInd = sims[proxy_inds].argmax()  # obtain per-camera max
                all_cam_tops.append(proxy_inds[maxInd])

            # find the top-K inds among the per-camera max
            all_cam_tops = torch.tensor(all_cam_tops)
            sel_ind = torch.argsort(sims[all_cam_tops])[-self.posK:]
            sims[all_cam_tops[sel_ind]] = 10000  # mask positive proxies
            top_inds = torch.sort(sims)[1][-self.bg_knn-self.posK:]
            sel_input = inputs[i, top_inds]
            loss += -1.0 * (F.log_softmax(sel_input.unsqueeze(0), dim=1) * soft_target.unsqueeze(0)).sum()

        loss /= len(inputs)
        return loss
