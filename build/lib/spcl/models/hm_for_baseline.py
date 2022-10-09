from __future__ import print_function, absolute_import
import torch
import torch.nn.functional as F
from torch import nn, autograd
from torch.autograd import Variable, Function
import numpy as np
import math
import collections
#from .contrast_loss import NegativeLoss, MemoTripletLoss
#from .cross_entropy_loss import CrossEntropyLabelSmooth

torch.autograd.set_detect_anomaly(True)

class ExemplarMemory(Function):
    def __init__(self, em, alpha=0.01, mixup_lam=1.0, hard_mining=False, weighted_mining=False):
        super(ExemplarMemory, self).__init__()
        self.em = em
        self.alpha = alpha
        self.mixup_lam = mixup_lam
        self.hard_mining = hard_mining
        self.weighted_mining = weighted_mining

    def forward(self, inputs, targets):
        self.save_for_backward(inputs*self.mixup_lam, targets)
        outputs = inputs.mm(self.em.t())
        return outputs

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(self.em)

        if self.hard_mining:
            batch_centers = collections.defaultdict(list) # a dict
            for feat, index in zip(inputs, targets.tolist()):
                batch_centers[index].append(feat.unsqueeze(0))
            for y in batch_centers.keys():
                batch_centers[y] = torch.cat(batch_centers[y])
                cls_sims = batch_centers[y].mm(self.em[y].unsqueeze(-1))
                hard_ind = torch.argmin(cls_sims)  # hard sample: the smallest similarity
                self.em[y] = self.alpha * self.em[y] + (1.0 - self.alpha) * batch_centers[y][hard_ind]
                self.em[y] /= self.em[y].norm()
        elif self.weighted_mining:
            batch_centers = collections.defaultdict(list)  # a dict
            for feat, index in zip(inputs, targets.tolist()):
                batch_centers[index].append(feat.unsqueeze(0))
            for y in batch_centers.keys():
                batch_centers[y] = torch.cat(batch_centers[y])
                cls_sims = batch_centers[y].mm(self.em[y].unsqueeze(-1))
                sorted_ind = torch.argsort(cls_sims.squeeze(), descending=True) # simiarity from high to low
                for ind in sorted_ind:
                    self.em[y] = self.alpha * self.em[y] + (1.0 - self.alpha) * batch_centers[y][ind]
                    self.em[y] /= self.em[y].norm()
        else:
            for x, y in zip(inputs, targets):
                self.em[y] = self.alpha * self.em[y] + (1.0 - self.alpha) * x
                self.em[y] /= self.em[y].norm()
        return grad_inputs, None


class HybridMemory(nn.Module):
    def __init__(self, temp=0.05, momentum=0.2, hard_mining=False, weighted_mining=False):
        super(HybridMemory, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.temp = temp
        self.momentum = momentum
        self.hard_mining = hard_mining
        self.weighted_mining = weighted_mining
        self.all_pseudo_label = ''
        self.features = ''

    def forward(self, inputs, targets):
        #mapped_targets = self.all_pseudo_label[targets].to(torch.device('cuda'))
        inputs = ExemplarMemory(self.features, alpha=self.momentum, hard_mining=self.hard_mining, weighted_mining=self.weighted_mining)(inputs, targets)
        inputs /= self.temp  # similarity score before softmax
        loss = F.cross_entropy(inputs, targets)

        return loss


class CameraAwareMemory(nn.Module):
    def __init__(self, temp=0.05, momentum=0.2, all_img_cams=''):
        super(CameraAwareMemory, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.temp = temp
        self.momentum = momentum
        self.all_img_cams = all_img_cams
        self.unique_cams = torch.unique(self.all_img_cams)
        self.pseudo_labels = ''
        self.features = ''
        self.memory_class_mapper = ''
        self.concate_intra_class = ''

    def forward(self, inputs, targets, cams, epoch):
        loss = torch.tensor([0.]).to(self.device)
        #print('batch index_target= {}'.format(targets))
        #print('batch cams= {}'.format(cams))
        if epoch >= 5:
            percam_tempV = []
            for ii in self.unique_cams:
                percam_tempV.append(self.features[ii].detach().clone())
            percam_tempV = torch.cat(percam_tempV, dim=0).to(torch.device('cuda'))

        for cc in torch.unique(cams):
            inds = torch.nonzero(cams == cc).squeeze(-1)
            percam_targets = self.pseudo_labels[targets[inds]]
            percam_feat = inputs[inds]
            mapped_targets = [self.memory_class_mapper[cc][int(k)] for k in percam_targets]
            mapped_targets = torch.tensor(mapped_targets).to(torch.device('cuda'))

            percam_inputs = ExemplarMemory(self.features[cc], alpha=self.momentum)(percam_feat, mapped_targets)
            percam_inputs /= self.temp  # similarity score before softmax
            loss += F.cross_entropy(percam_inputs, mapped_targets)

            # global contrastive loss
            if epoch >= 5:
                associate_loss = 0
                bg_knn = 50
                target_inputs = percam_feat.mm(percam_tempV.t().clone())
                temp_sims = target_inputs.detach().clone()
                target_inputs /= self.temp
                for k in range(len(percam_feat)):
                    ori_asso_ind = torch.nonzero(self.concate_intra_class == percam_targets[k]).squeeze(-1)
                    temp_sims[k, ori_asso_ind] = -1000.0  # mask out positive
                    sel_ind = torch.sort(temp_sims[k])[1][-bg_knn:]
                    concated_input = torch.cat((target_inputs[k, ori_asso_ind], target_inputs[k, sel_ind]), dim=0)
                    concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).cuda()
                    concated_target[0:len(ori_asso_ind)] = 1.0 / len(ori_asso_ind)
                    associate_loss += -1*(F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
                loss += 0.5 * associate_loss / len(percam_feat)
        return loss

