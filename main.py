import logging
import os
from datetime import datetime
import argparse

import json

import numpy as np
import torch
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data import CarDataset_train, CarDataset_test
from loss import ClassifyLoss
from model import BaseNet
from utils.utils import may_mkdir
from utils.utils import time_str
from utils.utils import str2bool

from sklearn.cluster import KMeans


class Config(object):
    def __init__(self):
        
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--sys_device_ids', type=str, default='0')
        parser.add_argument('--set_seed', type=str2bool, default=False)
        ## dataset parameter
        parser.add_argument('--split', type=int, default=8000)
        parser.add_argument('--resize', type=eval, default=(224, 224))
        parser.add_argument('--mirror', type=str2bool, default=True)
        parser.add_argument('--test-batch', default=4, type=int, help="has to be 1")
        parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
        parser.add_argument('--workers', type=int, default=4)
        # model
        parser.add_argument('-m', '--model', type=str, default='BASE',
                    choices=['BASE'])   

        parser.add_argument('--num_k', type=int, default = 5)
        # utils
        parser.add_argument('--evaluate', default=False, action='store_true', help="evaluation only")
        parser.add_argument('--save-dir', type=str, default='./saved_module')
        parser.add_argument('--use-cpu', action='store_true', help="use cpu")
        #train

        parser.add_argument('--steps_per_log', type=int, default=20)
        parser.add_argument('--epochs_per_val', type=int, default=1)
        parser.add_argument('--epochs_per_save', type=int, default=50)

        parser.add_argument('--max-epoch', default=30, type=int,
                    help="maximum epochs to run")
        parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
        parser.add_argument('--class-num', default=196, type=int,
                    help="number of classes")
        parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
        parser.add_argument('--stepsize', default=100, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
        parser.add_argument('--gamma', default=0.3, type=float,
                    help="learning rate decay")
        parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
        self.args = args = parser.parse_args()
        
        # gpu ids
        self.sys_device_ids = args.sys_device_ids
        # random
        self.set_seed = args.set_seed
        if self.set_seed:
            self.rand_seed = 0
        else: 
            self.rand_seed = None
        self.split = args.split
        self.resize = args.resize
        self.mirror = args.mirror
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.workers = args.workers
        self.model = args.model

        self.steps_per_log = args.steps_per_log
        self.epochs_per_val = args.epochs_per_val
        self.epochs_per_save = args.epochs_per_save

        # utils

        self.evaluate = args.evaluate
        self.save_dir = args.save_dir
        self.train_batch = args.train_batch
        self.test_batch = args.test_batch

        self.max_epoch = args.max_epoch
        self.start_epoch = args.start_epoch
        self.class_num = args.class_num
        self.lr = args.lr
        self.setpsize = args.stepsize
        self.gamma = args.gamma
        self.weight_decay = args.weight_decay 

        self.class_num = args.class_num

        # for model

### main function ###
cfg = Config()

optim_collection = {'Adam': optim.Adam, 'SGD': optim.SGD}

class Manager(object):

    def __init__(self, inf, cfg):
        self.cfg = cfg
        lr = cfg.lr
        self.model = model = cfg.model
        self.information = inf
        self.class_num = cfg.class_num

        normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)
        transform = transforms.Compose([
            transforms.Resize(cfg.resize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # 3*H*W, [0, 1]
            normalize,]) # normalize with mean/std
        # by a subset of attributes 
        train_set = CarDataset_train(
            cfg.split, 
            train=True, 
            transform = transform)

        train_loader = torch.utils.data.DataLoader(
            dataset = train_set,
            batch_size = cfg.train_batch,
            shuffle = True,
            num_workers = cfg.workers,
            pin_memory = True,
            drop_last = False)

        transform = transforms.Compose([
            transforms.Resize(cfg.resize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # 3*H*W, [0, 1]
            normalize,]) # normalize with mean/std
        # by a subset of attributes 
        val_set = CarDataset_train(
            cfg.split, 
            train=False, 
            transform = transform)

        val_loader = torch.utils.data.DataLoader(
            dataset = train_set,
            batch_size = cfg.train_batch,
            shuffle = True,
            num_workers = cfg.workers,
            pin_memory = True,
            drop_last = False)


        test_transform = transforms.Compose([
            transforms.Resize(cfg.resize),
            transforms.ToTensor(),
            normalize,])
        test_set = CarDataset_test(
            transform = test_transform)
        
        test_loader = torch.utils.data.DataLoader(
            dataset = test_set,
            batch_size = cfg.test_batch,
            shuffle = False,
            num_workers = cfg.workers,
            drop_last = False
        )
        self.train_data = train_set
        self.test_data = test_set 
        self.val_data = val_set 

        self.dataloader_train = train_loader
        self.dataloader_test = test_loader
        self.dataloader_val = val_loader
        if model=='BASE':
            self.net = BaseNet(self.class_num)
        
        self.net.cuda()
        self.optimizer = optim_collection['Adam'](params=self.net.parameters(), lr=lr)

        #self.criterion = criterion_collection['BCE']()
        self.criterion = ClassifyLoss(self.class_num)
        self.criterion = self.criterion.cuda()

        self.save_rootpath = cfg.save_dir
        self.save_module_path = cfg.save_dir
        path = self.mkdir_save(self.save_rootpath)
        logging.basicConfig(filename=path + '/log.log', filemode='a', level=logging.INFO)

        self.epoch = -1
        self.save_epoch = -1
        self.log_mark = False

    def train(self):
        train_len = len(self.dataloader_train)
        for i in range(self.cfg.start_epoch, self.cfg.max_epoch):
            self.train_framework_epoch()
            torch.cuda.empty_cache()
            if i % self.cfg.epochs_per_val==0 :
                self.eval(i)
                self.test(i)
            torch.cuda.empty_cache()

    def train_framework_epoch(self):
        self.net.train()
        for j, (img, labels) in enumerate(self.dataloader_train):
            #print("yjz_debug:labels:", labels)
            # if j ==0: break
            # print(datetime.now(),j)
            img = Variable(img.cuda())
            labels = Variable(labels.cuda())
            self.optimizer.zero_grad()
            x1 = self.net(img)

            loss_tmp = self.criterion(x1, labels)
            loss_tmp.backward(retain_graph=0)
            #retain_graph = len(outs)
            #loss_map = self.mapcriterion(target_map, using_map)
            #loss_map.backward(retain_graph = retain_graph)
            self.optimizer.step()
            print('success 1')
            if j % 100 == 0:
                print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), j)
            # if j == 2:
            #     break

    def eval(self, epoch):
        self.net.eval()
        for i, (img, labels) in enumerate(self.dataloader_val):
            #print("yjz_debug:labels:", labels)
            img = Variable(img.cuda())
            x1= self.net(img)
            pre_x1 =(torch.argmax(x1, dim=1)).cpu().detach().numpy()

            #pre_rec = (torch.sigmoid(rec) > 0.5).cpu().detach().numpy()
            if i == 0:
                pre_x1_arr = pre_x1.copy()
                #pre_rec_arr = pre_rec.copy()

            else:

                pre_x1_arr = np.hstack((pre_x1_arr, pre_x1))
                #pre_rec_arr = np.vstack((pre_rec_arr, pre_rec))

            labels = labels.cpu().numpy()
            if i == 0:
                val_label_arr = labels.copy()
            else:
                val_label_arr = np.hstack((val_label_arr, labels))

        print('***************************eval******************************')

        self.log(epoch, val_label_arr, pre_x1_arr, 'x_1')
        #np.save("map.npy", pre_rec_arr)    

    def log(self, epoch, val_label_arr, val_pre_arr, info=''):
        from metric import calculate_accuracy
        acc = calculate_accuracy(val_label_arr, val_pre_arr)
        log_str = '%s\t%s epoch: %d\tacc: %.4f' % (
        info, str(datetime.now().strftime('%H:%M:%S')), epoch + 1,acc)
        print(log_str)
        logging.info(log_str)

        acc_dict = {
                        'acc': acc,
                    }
        save_name = info + '_' + str(epoch) + '_' + '_' + str(acc)
        path = self.mkdir_save(self.save_module_path)
        ori_path = path
        path_pre = os.path.join(ori_path, save_name + '_pre.npy')
        path_pre_json = os.path.join(ori_path, save_name + '_pre.json')
        images = self.val_data.image

        #print("yjz_debug:val_label_arr:", val_label_arr.shape)
        #print("yjz_debug:val_pre_arr:", val_pre_arr.shape)
        #print("yjz_debug:images:", len(images))
        #print("yjz_debug:classes:", val_classes_arr.shape )
        #print("yjz_debug:reindex:", len(reindex))
        #print("yjz_debug:", images[0])
        
        dic = {}
        #print("yjz_deug:images:", images.shape)
        for i, ima in enumerate(images):
            temp_dic = {}
            temp_dic['pre'] = val_pre_arr[i].tolist()
            temp_dic['label'] = val_label_arr[i].tolist()
            dic[ima] = temp_dic

        json_str = json.dumps(dic, indent=4)
        with open(path_pre_json, 'w') as json_file:
            json_file.write(json_str)

        np.save(path_pre, val_pre_arr)
        if self.save_epoch != epoch:
            self.save(acc_dict, save_name)
            self.save_epoch = epoch
    

    def test(self, epoch):
        self.net.eval()
        for i, (imgname, img) in enumerate(self.dataloader_test):
            #print("yjz_debug:labels:", labels)
            imgname = np.array(imgname)
            img = Variable(img.cuda())
            x1= self.net(img)
            pre_x1 =(torch.argmax(x1, dim=1)).cpu().detach().numpy()
            #pre_rec = (torch.sigmoid(rec) > 0.5).cpu().detach().numpy()
            if i == 0:
                pre_img_arr = imgname.copy()
                pre_x1_arr = pre_x1.copy()
                #pre_rec_arr = pre_rec.copy()

            else:
                pre_x1_arr = np.hstack((pre_x1_arr, pre_x1))
                pre_img_arr = np.hstack((pre_img_arr, imgname))
                #pre_rec_arr = np.vstack((pre_rec_arr, pre_rec))

        print('***************************test******************************')
        self.save_test(epoch, pre_img_arr, pre_x1_arr, 'x_1')
        #np.save("pre.npy", pre_x1_arr)
    
    def save_test(self, epoch, img, label, info=''):
        save_name = info + '_' + str(epoch) + '_'
        path = self.mkdir_save(self.save_module_path)
        ori_path = path
        path_pre = os.path.join(ori_path, save_name + '_pre.txt')
        
        with open(path_pre, 'w') as test_file:
            for i in range(len(img)):
                result = '{} {}\n'.format(img[i], label[i]+1)
                test_file.write(result)
            test_file.close()


    def mkdir_save(self, rootpath):
        if not os.path.exists(rootpath):
            os.mkdir(rootpath)
        path = os.path.join(rootpath, datetime.now().strftime("%Y_%m_%d"))

        if not os.path.exists(path):
            os.mkdir(path)

        path = os.path.join(path, self.information)
        if not os.path.exists(path):
            os.mkdir(path)

        return path

    def save(self, acc_dict, save_name):

        path = self.mkdir_save(self.save_module_path)
        if not self.log_mark:
            logging.info(path)
            self.log_mark = True
        path = os.path.join(path, save_name + '.pt')
        print('save_path', path)
        logging.info('save_path: {}'.format(path))
        self.net.cpu()

        save_dict = {
            'acc_dict': acc_dict,
            'model_state_dict': self.net.state_dict(),
            'optim_state_dict': self.optimizer.state_dict()
        }
        self.net.cuda()
        torch.save(save_dict, path)

    def load(self, save_path):
        save_dict = torch.load(save_path)

        return save_dict


if __name__ == '__main__':
    print(cfg.args)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.sys_device_ids
    control = Manager(inf='base_train', cfg = cfg)
    #dic = control.load("model.pt")
    #control.net.load_state_dict(dic['model_state_dict'])
    control.eval(0)
    control.test(0)
    #control.train()
    #a = control.net.relation_part.all_map.cpu().detach().numpy()
    #np.save("all_map.npy", a)
    #a = control.net.relation_part.hier_map.cpu().detach().numpy()
    #np.save("hier_map.npy", a)
    if not(cfg.evaluate):
        labels = control.train()
        control.test(100)
    
    #control.test_perf()
    #a = control.net.relation_part.all_map.cpu().detach().numpy()
    #np.save("map.npy", a)