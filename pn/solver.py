import torch
from collections import OrderedDict
from torch.nn import utils, functional as F
from torch.optim import Adam,SGD
from torch.autograd import Variable
from torch.backends import cudnn
from networks.poolnet import build_model, weights_init
import scipy.misc as sm
import numpy as np
import os
import torchvision.utils as vutils
import cv2
import math
import time
from torchsnooper import snoop
def expected_time(cur_epoch,tot_epoch,cur_batchid,tot_batch,begin_time,cur_time):
    hav_batch=cur_epoch*tot_batch+cur_batchid
    if hav_batch>0:
        average_speed=(cur_time-begin_time)/hav_batch
        expected_finish_epoch=(tot_batch-cur_batchid)*average_speed
        expected_finish_all=(tot_batch-cur_batchid+(tot_epoch-cur_epoch-1)*tot_batch)*average_speed
        print('expecting to finish this epoch in {} hours, all tasks in {} hours'.format(expected_finish_epoch/3600,expected_finish_all/3600))
class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        self.lr_decay_epoch = [5,10,15,20]
        self.build_model()
        if config.mode == 'test':
            # print('Loading pre-trained model from %s...' % self.config.model)
            self.config.batch_size=1
            if self.config.cuda:
                self.net.load_state_dict(torch.load(self.config.model))
            else:
                self.net.load_state_dict(torch.load(self.config.model, map_location='cpu'))
            self.net.eval()

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))
        a=model.state_dict().keys()
        w=[]
        for i in a:
            w.append([i,model.state_dict()[i].numel()])
        w.sort(key=lambda t:t[1],reverse=True)
        other_tot=0
        backbone=0
        for i in w:
            if i[0].startswith('base.resnet'):backbone+=i[1]
            else:
                print(i)
                other_tot+=i[1]
        print('other tot = ',other_tot,'backbone = ',backbone,'tot_parameters = ',other_tot+backbone)
    # build the network
    def build_model(self):
        self.net = build_model(self.config.arch)
        if self.config.cuda:
            self.net = self.net.cuda()
        # self.net.train()
        self.net.eval()  # use_global_stats = True
        self.net.apply(weights_init)
        if self.config.load == '':
            self.net.base.load_pretrained_model(torch.load(self.config.pretrained_model))
        else:
            self.net.load_state_dict(torch.load(self.config.load))

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr, weight_decay=self.wd,momentum=0.9)
        if self.config.mode=='train':
            self.print_network(self.net, 'PoolNet Structure')

    def tr(self):
        for i, data_batch in enumerate(self.train_loader):
            print(i)
        print(self.train_loader.count_size)

    def test(self):
        mode_name = 'sal_fuse'
        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, name, im_size = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size'])
            with torch.no_grad():
                images = Variable(images)
                if self.config.cuda:
                    images = images.cuda()
                preds = self.net(images)
                pred = np.squeeze(torch.sigmoid(preds).cpu().data.numpy())
                multi_fuse = 255 * pred
                cv2.imwrite(os.path.join(self.config.test_fold, name[:-4] + '_' + mode_name + '.png'), multi_fuse)
                if (i+1)%50==0:
                    print(i+1,time.time()-time_s)
        time_e = time.time()
        print('Speed: %f FPS' % (img_num/(time_e-time_s)))
        print('Test Done!')

    # training phase
    # @snoop()
    def train(self):
        import sys
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        aveGrad = 0
        start_time=time.time()
        import matplotlib.pyplot as plt
        x=[]
        sc=[]
        for epoch in range(self.config.epoch):
            r_sal_loss= 0
            self.net.zero_grad()

            for i, data_batch in enumerate(self.train_loader):
                cur_id=i+epoch*iter_num
                x.append(cur_id)
                sal_image, sal_label = data_batch['sal_image'], data_batch['sal_label']
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                sal_image, sal_label= Variable(sal_image), Variable(sal_label)
                if self.config.cuda:
                    # cudnn.benchmark = True
                    sal_image, sal_label = sal_image.cuda(), sal_label.cuda()

                sal_pred = self.net(sal_image)
                
                sal_loss_fuse = F.binary_cross_entropy_with_logits(sal_pred, sal_label, reduction='sum')
                sal_loss = sal_loss_fuse / (self.iter_size * self.config.batch_size)

                sc.append(sal_loss.item())
                r_sal_loss += sal_loss.data
                sal_loss.backward()

                aveGrad += 1

                # accumulate gradients as done in DSS
                if aveGrad % self.iter_size == 0:
#                     引入clip
                    import torch.nn as nn
                    nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=20, norm_type=2)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    aveGrad = 0

                if i % (self.show_every // self.config.batch_size) == 0:
                    if i == 0:
                        x_showEvery = 1
                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Sal : %10.4f' % (
                        epoch, self.config.epoch, i, iter_num, r_sal_loss/x_showEvery))
                    print('Learning rate: ' + str(self.lr))
                    r_sal_loss= 0
                    
                if i%50==0:
                    expected_time(epoch, self.config.epoch, i, iter_num, start_time, time.time())
                # if i%1000==0:
                #     torch.save(self.net.state_dict(), '%s/models/epoch_%d_step_%d.pth' % (self.config.save_folder, epoch + 1,i))

            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/models/epoch_%d.pth' % (self.config.save_folder, epoch + 1))
                
                from test_model import test_model
                from main import get_last_runid
                test_model(get_last_runid(),'epoch_{}'.format(epoch+1))
                
                
            plt.plot(x,sc)
            plt.savefig('%s/models/epoch_%d.png'% (self.config.save_folder, epoch + 1))
            if epoch in self.lr_decay_epoch:
                self.lr = self.lr * 0.1
                self.optimizer = SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr, weight_decay=self.wd,momentum=0.9)

        torch.save(self.net.state_dict(), '%s/models/final.pth' % self.config.save_folder)

def bce2d(input, target, reduction=None):
    assert(input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg  / num_total
    beta = 1.1 * num_pos  / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)

