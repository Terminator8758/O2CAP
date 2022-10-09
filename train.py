from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import copy
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from reid import datasets
from reid import models
from reid.models.o2cap import CameraAwareMemory
from reid import trainers
from reid.evaluators import Evaluator, extract_features
from reid.utils.data import IterLoader
from reid.utils.data import transforms as T
from reid.utils.data.sampler import ClassUniformlySampler
from reid.utils.data.preprocessor import Preprocessor, CameraAwarePreprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.utils.faiss_rerank import compute_jaccard_distance
from bisect import bisect_right


start_epoch = best_mAP = 0

def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    print('root path= {}'.format(root))
    dataset = datasets.create(name, root)
    return dataset

def get_train_loader(args, dataset, height, width, batch_size, workers,
                    num_instances, iters, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
             T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
         ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = ClassUniformlySampler(train_set, class_position=4, k=num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
                DataLoader(CameraAwarePreprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500,
                 warmup_method="linear", last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,)

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / float(self.warmup_iters)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

def create_model(args):
    if args.arch=='resnet50':
        model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout, num_classes=0, pool_type=args.pool_type)
    elif args.arch=='resnet50_ibn':
        model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout, num_classes=0, pool_type=args.pool_type)

    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model


def main(args):
    #args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'train_log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters>0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)

    # create image-level camera information
    all_img_cams = torch.tensor([c for _, _, c in sorted(dataset.train)])
    temp_all_cams = all_img_cams.numpy()
    all_img_cams = all_img_cams.cuda()
    unique_cameras = torch.unique(all_img_cams)

    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model = create_model(args)

    # Create memory
    memory = CameraAwareMemory(temp=args.temp, momentum=args.momentum, all_img_cams=all_img_cams,
                               has_cross_cam_loss=True, has_online_proxy_loss=True, posK=3).cuda()
 
    # get propagate loader
    cluster_loader = get_test_loader(dataset, args.height, args.width,
                                     args.batch_size, args.workers, testset=sorted(dataset.train))
    
    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    milestones = [20, 40]
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=0.1, warmup_factor=0.01, warmup_iters=10)

    # Trainer
    trainer = trainers.CAPTrainer_USL(model, memory)  # note: trainer needs to change as training setting changes.

    for epoch in range(args.epochs):
        # Calculate distance
        print('==> Create pseudo labels for unlabeled data with self-paced policy')
        # features = memory.features.clone()
        features, _ = extract_features(model, cluster_loader, print_freq=100)
        features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
        rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)

        if (epoch==0):
            # DBSCAN cluster
            eps = args.eps
            print('Clustering criterion: eps: {:.3f}'.format(eps))
            cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

        # select & cluster images as training set of this epochs
        pseudo_labels = cluster.fit_predict(rerank_dist)
        num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

        # generate proxy labels (camera-aware sub-cluster label)
        proxy_labels = -1 * np.ones(pseudo_labels.shape, pseudo_labels.dtype)
        cnt = 0
        for i in range(0, int(pseudo_labels.max() + 1)):
            inds = np.where(pseudo_labels == i)[0]
            local_cams = temp_all_cams[inds]
            for cc in np.unique(local_cams):
                pc_inds = np.where(local_cams==cc)[0]
                proxy_labels[inds[pc_inds]] = cnt
                cnt += 1
        num_proxies = len(set(proxy_labels)) - (1 if -1 in proxy_labels else 0)

        # generate new dataset and calculate cluster centers
        pseudo_labeled_dataset = []
        for i, ((fname, _, cid), label, accum_y) in enumerate(zip(sorted(dataset.train), pseudo_labels, proxy_labels)):
            if label != -1:
                pseudo_labeled_dataset.append((fname, label, cid, i, accum_y)) 

        # statistics of clusters and un-clustered instances
        outlier_num = len(np.where(pseudo_labels==-1)[0])
        print('==> Statistics for epoch {}: {} clusters, {} sub-clusters, {} un-clustered instances'.
              format(epoch, num_ids, num_proxies, outlier_num))

        # re-initialize memory (pseudo label, memory feature and others)
        pseudo_labels = torch.from_numpy(pseudo_labels).long()
        memory.all_pseudo_label = pseudo_labels.cuda()
        proxy_labels = torch.from_numpy(proxy_labels).long()
        memory.all_proxy_label = proxy_labels.cuda()

        memory.proxy_label_dict = {}  # {pseudo_label1: [proxy3, proxy10],...}
        for c in range(0, int(memory.all_pseudo_label.max() + 1)):
            memory.proxy_label_dict[c] = torch.unique(memory.all_proxy_label[memory.all_pseudo_label == c])

        memory.proxy_cam_dict = {}  # for computing proxy enhance loss
        for cc in unique_cameras:
            proxy_inds = torch.unique(memory.all_proxy_label[(all_img_cams == cc) & (memory.all_proxy_label>=0)])
            memory.proxy_cam_dict[int(cc)] = proxy_inds

        ####################################
        ## re-initialize cluster memory
        #cluster_centers = torch.zeros(num_ids, features.size(1))
        #for ii in range(num_ids):
        #    idx = torch.nonzero(pseudo_labels == ii).squeeze(-1)
        #    cluster_centers[ii] = features[idx].mean(0)
        #cluster_centers = F.normalize(cluster_centers.detach(), dim=1).cuda()
        #print('  initializing cluster memory feature with shape {}...'.format(cluster_centers.shape))
        #memory.cluster_memory = cluster_centers.detach()

        # initialize proxy memory
        proxy_centers = torch.zeros(num_proxies, features.size(1))
        for lbl in range(num_proxies):
            ind = torch.nonzero(proxy_labels == lbl).squeeze(-1)  # note here
            id_feat = features[ind].mean(0)
            proxy_centers[lbl,:] = id_feat
        proxy_centers = F.normalize(proxy_centers.detach(), dim=1).cuda()
        print('  initializing proxy memory feature with shape {}...'.format(proxy_centers.shape))
        memory.global_memory = proxy_centers.detach()
        ####################################

        # update train loader and train an epoch
        train_loader = get_train_loader(args, dataset, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset)

        train_loader.new_epoch()

        trainer.train(epoch, train_loader, optimizer,
                    print_freq=args.print_freq, train_iters=len(train_loader))

        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
            print('==> Epoch {} test: '.format(epoch))
            evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

        lr_scheduler.step()

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CAP enhancement for unsupervised re-ID")
    # data
    parser.add_argument('--dataset', type=str, default='Market1501',
                        choices=datasets.names())
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num_instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.5,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--k1', type=int, default=20,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50')
    parser.add_argument('-pool_type', type=str, default='avgpool')
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--eval-step', type=int, default=10)
    parser.add_argument('--temp', type=float, default=0.07,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH',
                        default='/path/to/dataset/')
    parser.add_argument('--logs_dir', type=str, metavar='PATH',
                        default='train_logs/')
    args = parser.parse_args()    

    main(args)

