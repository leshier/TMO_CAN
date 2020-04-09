import os
import time
from functools import reduce
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
import np_transforms
from E2etmo import E2ETMO
from ImageDataset import ImageDataset
from NLPD import NLPD_Loss
from Gdn import Gdn2d, Gdn1d
from PIL import Image
import skimage.color
import time
import numpy
import skimage.color as color

from torchvision import utils as vutils
import cv2
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
import scipy.io as io
import matplotlib
from torch import nn

class Trainer(object):
    def __init__(self, config):
        torch.manual_seed(config.seed)

        self.train_batch_size = config.batch_size
        self.test_batch_size = 1
        self.results_savepath = config.results_savepath

        self.train_transform = np_transforms.Compose([
            np_transforms.RandomCrop(config.image_size),
            np_transforms.RandomHorizontalFlip(),
            np_transforms.ToTensor()
        ])

        self.test_transform = transforms.Compose([
            np_transforms.ToTensor()
        ])

        self.train_data = ImageDataset(csv_file=os.path.join(config.trainset, 'train.txt'),
                                       img_dir=config.trainset,
                                       transform=self.train_transform,
                                       test=False)

        self.train_loader = DataLoader(self.train_data,
                                       batch_size=self.train_batch_size,
                                       shuffle=True,
                                       pin_memory=True,
                                       num_workers=4)

        # testing set configuration
        self.test_data = ImageDataset(csv_file=os.path.join(config.testset, 'tests.txt'),
                                      img_dir=config.testset,
                                      transform=self.test_transform,
                                      test=True)

        self.test_loader = DataLoader(self.test_data,
                                      batch_size=self.test_batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=4,
                                      )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() and config.use_cuda else "cpu")

        self.model = E2EPOIR(config.layer)
        self.model.to(self.device)
       

        self.model_name = type(self.model).__name__
        # print(self.model)

        # loss function
        self.loss_fn = NLPD_Loss()
        self.loss_fn.to(self.device)
        self.initial_lr = config.lr
        if self.initial_lr is None:
            lr = 0.0005
        else:
            lr = self.initial_lr

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=lr,
                                    weight_decay=1e-4)


        self.start_epoch = 0
        self.start_step = 0
        self.train_loss = []
        self.test_results = []
        self.ckpt_path = config.ckpt_path
        self.max_epochs = config.max_epochs
        self.epochs_per_eval = config.epochs_per_eval
        self.epochs_per_save = config.epochs_per_save

        # try load the model
        if config.resume or not config.train:
            if config.ckpt:
                ckpt = os.path.join(config.ckpt_path, config.ckpt)
            else:
                ckpt = self._get_latest_checkpoint(path=config.ckpt_path)
            if not ckpt:
                pass
            else:
                self._load_checkpoint(ckpt=ckpt)

        self.scheduler = lr_scheduler.StepLR(self.optimizer,
                                             last_epoch=self.start_epoch-1,
                                             step_size=config.decay_interval,
                                             gamma=config.decay_ratio)


    def fit(self):
        for epoch in range(self.start_epoch, self.max_epochs):
            self.epoch_loss = []
            _ = self._train_single_epoch()
            #print(123)
        writer.close()

    def _train_single_epoch(self):

        # initialize logging system
        num_steps_per_epoch = len(self.train_loader)
        local_counter = epoch * num_steps_per_epoch + 1
        start_time = time.time()
        beta = 0.9
        running_loss = 0 if epoch == 0 else self.train_loss[-1]
        loss_corrected = 0.0
        running_duration = 0.0

        # start training
        print('[*] Adam learning rate: {:f}'.format(self.optimizer.param_groups[0]['lr']))
        for step, sample_batched in enumerate(self.train_loader, 0):
            if step < self.start_step:
                continue
            self.model.train()
            x = sample_batched['nlp_I']
            hdr = sample_batched['simulate_L']

            for index in range(x.__len__()):
                x[index] = x[index].to(self.device)
            hdr = hdr.to(self.device)
            y = self.model(x)

            self.optimizer.zero_grad()

            self.loss = self.loss_fn(hdr, y)
            self.loss.backward()
            self.optimizer.step()

            self._gdn_param_proc()

            # statistics
            running_loss = beta * running_loss + (1 - beta) * self.loss.data.item()
            loss_corrected = running_loss / (1 - beta ** local_counter)

            current_time = time.time()
            duration = current_time - start_time
            running_duration = beta * running_duration + (1 - beta) * duration
            duration_corrected = running_duration / (1 - beta ** local_counter)
            examples_per_sec = self.train_batch_size / duration_corrected
            format_str = ('(E:%d, S:%d) [Loss = %.4f] (%.1f samples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (epoch, step, self.loss,
                                examples_per_sec, duration_corrected))

            local_counter += 1
            self.start_step = 0
            start_time = time.time()
            self.epoch_loss.append(float(self.loss.data.cpu().data))
        self.train_loss.append(loss_corrected)
        self.scheduler.step()

        if (epoch+1) % self.epochs_per_eval == 0:
            # evaluate after every other epoch
            self.model.eval()
            test_results = self.eval(epoch)
            self.test_results.append(test_results)
            out_str = 'Epoch {} Testing: NLPD: {:.4f}'.format(epoch, test_results)
            #writer.add_scalar('Test/Loss', test_results, epoch)
            print(out_str)

        if (epoch+1) % self.epochs_per_save == 0:
            model_name = '{}-{:0>5d}.pt'.format(self.model_name, epoch)
            model_name = os.path.join(self.ckpt_path, model_name)
            self._save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'train_loss': self.train_loss,
                'test_results': self.test_results,
            }, model_name)

        return self.loss.data.item()

    def eval(self):
        nlpd_score = []

        for step, sample_batched in enumerate(self.test_loader, 0):

            x = sample_batched['nlp_I']

            hdr = sample_batched['simulate_L']

            hdr_name = sample_batched['hdr_name']

            hdr_h = sample_batched['hdr_h']

            start_time = time.time()
            for index in range(x.__len__()):
                x[index] = x[index].to(self.device)
            hdr = hdr.to(self.device)
            y = self.model(x)
            stop_time = time.time()
            self._save_image(y, self.results_savepath, hdr_name[0], hdr_h)
            loss = self.loss_fn(hdr, y)

            print(hdr_name[0]+' '+str(float(loss)) + ' ' + str(stop_time-start_time))
            nlpd_score.append(float(loss.data.cpu().data))
        return reduce(lambda l1, l2: l1 + l2, nlpd_score) / len(nlpd_score)

    def _gdn_param_proc(self):
        for m in self.model.modules():
            if isinstance(m, Gdn2d) or isinstance(m, Gdn1d):
                m.beta.data.clamp_(min=2e-10)
                m.gamma.data.clamp_(min=2e-10)
                m.gamma.data = (m.gamma.data + m.gamma.data.t()) / 2


    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt)
            self.start_epoch = checkpoint['epoch']+1
            self.train_loss = checkpoint['train_loss']
            self.test_results = checkpoint['test_results']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.initial_lr is not None:
                for param_group in self.optimizer.param_groups:
                    param_group['initial_lr'] = self.initial_lr
            print("[*] loaded checkpoint '{}' (epoch {})"
                  .format(ckpt, checkpoint['epoch']))
        else:
            print("[!] no checkpoint found at '{}'".format(ckpt))

    @staticmethod
    def _get_latest_checkpoint(path):
        ckpts = os.listdir(path)
        ckpts = [ckpt for ckpt in ckpts if not os.path.isdir(os.path.join(path, ckpt))]
        all_times = sorted(ckpts, reverse=True)
        if len(all_times) == 0 :
            print('[*] No found checkpoint')
            return False
        else:
            return os.path.join(path, all_times[0])

    # save checkpoint
    @staticmethod
    def _save_checkpoint(state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)



    def save_image_tensor(self,input_tensor: torch.Tensor, filename):
     
        assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    
        input_tensor = input_tensor.clone().detach()
     
        input_tensor = input_tensor.to(torch.device('cpu'))
       
        
        vutils.save_image(input_tensor, filename)

    def _save_image(self, img, path, name, hdr_h):
        # color
        self.d_max = 500
        self.d_min = 5
        t = img.data[0].cpu()
        t[t > self.d_max] = self.d_max
        t[t < self.d_min] = self.d_min
        t = (t - self.d_min) / (self.d_max - self.d_min)
        t = (t ** (1 / 2.2))

        hdr_h = hdr_h.squeeze().cpu().numpy()
        hdr_h[:,:,2] = t.squeeze().cpu().numpy()
        hdr_h[:,:,1] = hdr_h[:,:,1]

        result = color.hsv2rgb(hdr_h)

        plt.imsave(path+name[-10:]+'.png',result)






