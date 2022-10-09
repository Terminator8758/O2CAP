from __future__ import print_function, absolute_import
import torch
import torch.nn.functional as F
from torch import nn, autograd
from torch.autograd import Variable, Function
import numpy as np


torch.autograd.set_detect_anomaly(True)

class ExemplarMemory(Function):
    def __init__(self, em, alpha=0.01):
        super(ExemplarMemory, self).__init__()
        self.em = em
        self.alpha = alpha

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs = inputs.mm(self.em.t())
        return outputs

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(self.em)
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
        # mapped_targets = self.all_pseudo_label[targets].to(torch.device('cuda'))
        inputs = ExemplarMemory(self.features, alpha=self.momentum)(inputs, targets)
        inputs /= self.temp  # similarity score before softmax
        loss = F.cross_entropy(inputs, targets)

        return loss


class CameraAwareMemory(nn.Module):
    def __init__(self, temp=0.05, momentum=0.2, all_img_cams='', has_intra_proxy_loss=False, has_cross_proxy_loss=False,
                 has_proxy_enhance_loss=False, has_cluster_loss=False, joint_cluster_proxy=False, percam_loss=True, 
                 bg_knn=50, posK=3, cluster_temp=0.05, cluster_hard_mining=False, intra_proxy_rectify=False):
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
        self.features = ''
        self.memory_class_mapper = ''
        self.concate_intra_class = ''
        self.clusters = ''
        self.has_intra_proxy_loss = has_intra_proxy_loss
        self.has_cross_proxy_loss = has_cross_proxy_loss
        self.has_proxy_enhance_loss = has_proxy_enhance_loss
        self.has_cluster_loss = has_cluster_loss
        self.joint_cluster_proxy = joint_cluster_proxy
        self.percam_loss = percam_loss
        self.bg_knn = bg_knn
        self.posK = posK
        self.cluster_temp = cluster_temp
        self.cluster_hard_mining = cluster_hard_mining
        self.intra_proxy_rectify = intra_proxy_rectify
        # use of intra_proxy_rectify: when an instance's most similar intra-camera proxy is not the given proxy, do not use it to update memory

    def forward(self, features, targets, cams, epoch):

        pseudo_y = self.all_pseudo_label[targets].to(torch.device('cuda'))  # targets: image index in the train set
        proxy_targets = self.all_proxy_label[targets].to(torch.device('cuda'))
        loss = torch.tensor(0.).to(torch.device('cuda'))
 
        # cluster contrastive loss
        if self.has_cluster_loss:
            cluster_score = ExemplarMemory(self.cluster_memory, alpha=self.momentum)(features, pseudo_y)
            cluster_score /= self.cluster_temp
            #if not self.joint_cluster_proxy:
            #    loss += F.cross_entropy(cluster_score, pseudo_y)

        # proxy-level contrastive loss
        #print('batch index target= {}'.format(targets))
        #print('batch proxy target= {}'.format(proxy_targets))
        #print('batch cams= {}'.format(cams))
        score = ExemplarMemory(self.global_memory, alpha=self.momentum)(features, proxy_targets)
        inputs = score / self.temp  # similarity score before softmax
        intra_loss, associate_loss, enhance_loss = 0, 0, 0
        noise_cnt = 0

        if self.percam_loss:
            for cc in torch.unique(cams):
                inds = torch.nonzero(cams == cc).squeeze(-1)

                percam_inputs = inputs[inds]
                percam_y = pseudo_y[inds]
                pc_prx_target = proxy_targets[inds]
                temp_score = score[inds].detach().clone()  # for similarity
                
                # for checking:
                #intra_prxs = self.proxy_cam_dict[int(cc)]
                #max_prx = temp_score[:, intra_prxs].max(dim=1)[1]
                #noise_cnt += len(torch.nonzero(intra_prxs[max_prx] != pc_prx_target))
 
                if self.joint_cluster_proxy:
                    pc_cluster_score = cluster_score[inds]
                    associate_loss += self.get_joint_cluster_proxy_associate_loss(percam_inputs, percam_y, pc_cluster_score)

                elif self.has_cross_proxy_loss:
                    associate_loss += self.get_proxy_associate_loss(percam_inputs, percam_y)
 
                if self.has_proxy_enhance_loss:
                    enhance_loss += self.get_proxy_cam_wise_nn_enhance_loss(temp_score, percam_inputs, pc_prx_target, balance_w=0.05)
        else:
            if self.has_intra_proxy_loss:
                intra_loss += self.get_intra_loss_all_cams(inputs, proxy_targets, cams)

            if self.joint_cluster_proxy:
                associate_loss += self.get_joint_cluster_proxy_associate_loss(inputs, pseudo_y, cluster_score)
            elif self.has_cross_proxy_loss:
                associate_loss += self.get_proxy_associate_loss(inputs, pseudo_y)

            if self.has_proxy_enhance_loss:
                temp_score = score.detach().clone()
                enhance_loss += self.get_proxy_cam_wise_nn_enhance_loss(temp_score, inputs, proxy_targets, balance_w=0.05)

        #print('batch associate_los= {}, enhance_loss= {}'.format(associate_loss.detach().data, enhance_loss.detach().data))
        loss += (intra_loss + 1.0 * associate_loss + enhance_loss)
        #if noise_cnt > 0:
        #    print('batch noise cnt= ', noise_cnt)
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
            #print('instance positive proxies by clustering: ', pos_ind.detach().data)
            temp_inputs[i, pos_ind] = 10000.0  # mark the positives
            sel_ind = torch.sort(temp_inputs[i])[1][-self.bg_knn-len(pos_ind):]
            sel_input = inputs[i, sel_ind]
            sel_target = torch.zeros((len(sel_input)), dtype=sel_input.dtype).to(torch.device('cuda'))
            sel_target[-len(pos_ind):] = 1.0 / len(pos_ind)
            loss += -1.0 * (F.log_softmax(sel_input.unsqueeze(0), dim=1) * sel_target.unsqueeze(0)).sum()
        loss /= len(inputs)
        return loss


    def get_joint_cluster_proxy_associate_loss(self, inputs, targets, cluster_inputs):
        '''
        inputs: batch to proxies similarity after scaling
        cluster_inputs: batch to cluster similarity after scaling
        targets: batch pseudo label
        '''
        if self.cluster_hard_mining:
            temp_cluster = cluster_inputs.detach().clone()

        temp_inputs = inputs.detach().clone()
        loss = 0
        for i in range(len(inputs)):
            pos_ind = self.proxy_label_dict[int(targets[i])]
            temp_inputs[i, pos_ind] = 10000.0  # mark the positives
            sel_ind = torch.sort(temp_inputs[i])[1][-self.bg_knn-len(pos_ind):]
            if self.cluster_hard_mining:
                temp_cluster[i, targets[i]] = 10000.0
                cluster_ind = torch.sort(temp_cluster[i])[1][-self.bg_knn-1:]
                concated_input = torch.cat((cluster_inputs[i, cluster_ind], inputs[i, sel_ind]), dim=0)
                concated_target = torch.zeros((len(concated_input)), dtype=inputs.dtype).to(torch.device('cuda'))
                concated_target[self.bg_knn] = 1.0 / (len(pos_ind) + 1)
                concated_target[-len(pos_ind):] = 1.0 / (len(pos_ind) + 1)
            else:
                concated_input = torch.cat((cluster_inputs[i], inputs[i, sel_ind]), dim=0)
                concated_target = torch.zeros((len(concated_input)), dtype=inputs.dtype).to(torch.device('cuda'))
                concated_target[targets[i]] = 1.0 / (len(pos_ind) + 1)
                concated_target[-len(pos_ind):] = 1.0 / (len(pos_ind) + 1)
            loss += -1.0 * (F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
        loss /= len(inputs)
        return loss


    def get_proxy_cam_wise_nn_enhance_loss(self, temp_score, inputs, proxy_targets, balance_w=0.05):

        temp_memory = self.global_memory.detach().clone()  # global_memory is the proxy memory
        soft_target = torch.zeros(self.bg_knn + self.posK, dtype=torch.float).to(torch.device('cuda'))
        soft_target[-self.posK:] = 1.0 / self.posK
        loss = 0

        for i in range(len(inputs)):
            lbl = proxy_targets[i]
            sims = balance_w * temp_score[i] + (1 - balance_w) * torch.matmul(temp_memory[lbl], temp_memory.t())

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
