import numpy as np
import time

from tqdm import tqdm
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable

from agents.base import BaseAgent

# import your classes here
from graphs.models.unet import unet
from datasets.DroneDataset import DroneDataset

from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList
from utils.misc import print_cuda_statistics

cudnn.benchmark = False


def pixel_accuracy(output, mask):
    import torch.nn.functional as F
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy


def mIoU(pred_mask, mask, smooth=1e-10, n_classes=23):
    import torch.nn.functional as F
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class UNet(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        # define models
        self.model = unet(n_channels=3, n_classes=23)

        # define data_loader
        self.data_loader = DroneDataset

        # define loss
        self.loss = nn.CrossEntropyLoss()

        # define optimizers for both generator and discriminator
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed_all(self.manual_seed)
            torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()
            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)
        # Summary Writer
        self.summary_writer = None

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        pass

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        pass

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train_one_epoch()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    #def train(self):
    #    """
    #    Main training loop
    #    :return:
    #    """
    #    pass

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        torch.cuda.empty_cache()
        train_losses = []
        test_losses = []
        val_iou = []; val_acc = []
        train_iou = []; train_acc = []
        lrs = []
        min_loss = np.inf
        decrease = 1 ; not_improve=0

        self.model.to(self.config.gpu_device)
        fit_time = time.time()
        for e in range(self.config.max_epoch):
            since = time.time()
            running_loss = 0
            iou_score = 0
            accuracy = 0
            #training loop
            self.model.train()
            for i, data in enumerate(tqdm(train_loader)):
                #training phase
                image_tiles, mask_tiles = data
                
                image = image_tiles.to(self.config.gpu_device); mask = mask_tiles.to(self.config.gpu_device);
                #forward
                output = self.model(image)
                loss = self.loss(output, mask)
                #evaluation metrics
                iou_score += mIoU(output, mask)
                accuracy += pixel_accuracy(output, mask)
                #backward
                loss.backward()
                self.optimizer.step() #update weight          
                self.optimizer.zero_grad() #reset gradient
                
                #step the learning rate
                lrs.append(get_lr(self.optimizer))
                scheduler.step() 
                
                running_loss += loss.item()
                
            else:
                self.model.eval()
                test_loss = 0
                test_accuracy = 0
                val_iou_score = 0

                #validation loop
                with torch.no_grad():
                    for i, data in enumerate(tqdm(val_loader)):
                        #reshape to 9 patches from single image, delete batch size
                        image_tiles, mask_tiles = data
                        
                        image = image_tiles.to(self.config.gpu_device); mask = mask_tiles.to(self.config.gpu_device);
                        output = self.model(image)
                        #evaluation metrics
                        val_iou_score +=  mIoU(output, mask)
                        test_accuracy += pixel_accuracy(output, mask)
                        #loss
                        loss = self.loss(output, mask)                                  
                        test_loss += loss.item()
                
                #calculatio mean for each batch
                train_losses.append(running_loss/len(train_loader))
                test_losses.append(test_loss/len(val_loader))


                if min_loss > (test_loss/len(val_loader)):
                    print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, (test_loss/len(val_loader))))
                    min_loss = (test_loss/len(val_loader))
                    decrease += 1
                    if decrease % 5 == 0:
                        print('saving model...')
                        torch.save(self.model, 'Unet-Mobilenet_v2_mIoU-{:.3f}.pt'.format(val_iou_score/len(val_loader)))
                        

                if (test_loss/len(val_loader)) > min_loss:
                    not_improve += 1
                    min_loss = (test_loss/len(val_loader))
                    print(f'Loss Not Decrease for {not_improve} time')
                    if not_improve == 7:
                        print('Loss not decrease for 7 times, Stop Training')
                        break
                
                #iou
                val_iou.append(val_iou_score/len(val_loader))
                train_iou.append(iou_score/len(train_loader))
                train_acc.append(accuracy/len(train_loader))
                val_acc.append(test_accuracy/ len(val_loader))
                print("Epoch:{}/{}..".format(e+1, epochs),
                    "Train Loss: {:.3f}..".format(running_loss/len(train_loader)),
                    "Val Loss: {:.3f}..".format(test_loss/len(val_loader)),
                    "Train mIoU:{:.3f}..".format(iou_score/len(train_loader)),
                    "Val mIoU: {:.3f}..".format(val_iou_score/len(val_loader)),
                    "Train Acc:{:.3f}..".format(accuracy/len(train_loader)),
                    "Val Acc:{:.3f}..".format(test_accuracy/len(val_loader)),
                    "Time: {:.2f}m".format((time.time()-since)/60))
            
        history = {'train_loss' : train_losses, 'val_loss': test_losses,
                'train_miou' :train_iou, 'val_miou':val_iou,
                'train_acc' :train_acc, 'val_acc':val_acc,
                'lrs': lrs}
        print('Total time: {:.2f} m' .format((time.time()- fit_time)/60))
        return history

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        pass

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        pass
