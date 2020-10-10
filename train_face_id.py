import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import os
from torchvision import transforms
from models import RSCNN_SSN_Cls as RSCNN_SSN
from data import Bosphorus, BU3DFE
import utils.pytorch_utils as pt_utils
import utils.pointnet2_utils as pointnet2_utils
import data.data_utils as d_utils
import argparse
import random
import yaml
#import visdom

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)            
torch.cuda.manual_seed(seed)       
torch.cuda.manual_seed_all(seed) 
######################################################################################
"""
vis = visdom.Visdom(port='8098')
def plot_current_errors(plot_data): # plot_data: {'X':list, 'Y':list, 'legend':list}
    vis.line(
        X=np.stack([np.array(plot_data['X'])]*len(plot_data['legend']),1),
        Y=np.array(self.plot_data['Y']),
        opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)
plot_data = {'X':[], 'Y':[], 'legend':['train_loss']}
"""
######################################################################################
parser = argparse.ArgumentParser(description='Relation-Shape CNN Face ID Classification Training')
parser.add_argument('--config', default='cfgs/config_ssn_face_id.yaml', type=str)

def main():
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    print("\n**************************")
    for k, v in config['common'].items():
        setattr(args, k, v)
        print('\n[%s]:'%(k), v)
    print("\n**************************\n")
    
    try:
        os.makedirs(args.save_path)
    except OSError:
        pass
    
    train_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])
    test_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])
    
    train_dataset = BU3DFE(num_points=args.num_points, root=args.data_root, transforms=train_transforms, train=True, task='id')
    #train_dataset = Bosphorus(num_points=args.num_points, root=args.data_root, transforms=train_transforms, train=True, task='id')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers = int(args.workers),
        drop_last=False
    )

    test_dataset = BU3DFE(num_points=args.num_points, root=args.data_root, transforms=test_transforms, train=False, task='id')
    #test_dataset = Bosphorus(num_points=args.num_points, root=args.data_root, transforms=test_transforms, train=False, task='id')
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=int(args.workers),
        drop_last=False
    )
    
    model = RSCNN_SSN(num_classes=args.num_classes, input_channels = args.input_channels, relation_prior = args.relation_prior, use_xyz = True)
    model.cuda()
    optimizer = optim.Adam(
        model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    #optimizer = optim.SGD(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay, momentum=0.9)

    lr_lbmd = lambda e: max(args.lr_decay**(e // args.decay_step), args.lr_clip / args.base_lr)
    bnm_lmbd = lambda e: max(args.bn_momentum * args.bn_decay**(e // args.decay_step), args.bnm_clip)
    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd)
    bnm_scheduler = pt_utils.BNMomentumScheduler(model, bnm_lmbd)
    
    if args.checkpoint is not '':
        model.load_state_dict(torch.load(args.checkpoint))
        print('Load model successfully: %s' % (args.checkpoint))

    criterion = nn.CrossEntropyLoss()
    num_batch = len(train_dataset)/args.batch_size
    
    # training
    train(train_dataloader, test_dataloader, model, criterion, optimizer, lr_scheduler, bnm_scheduler, args, num_batch)
    

def train(train_dataloader, test_dataloader, model, criterion, optimizer, lr_scheduler, bnm_scheduler, args, num_batch):
    PointcloudScaleAndTranslate = d_utils.PointcloudScaleAndTranslate()   # initialize augmentation
    global g_acc 
    g_acc = 0.94    # only save the model whose acc > 0.92
    batch_count = 0
    model.train()
    for epoch in range(args.epochs):
        
        for i, data in enumerate(train_dataloader, 0):
            if lr_scheduler is not None:
                lr_scheduler.step(epoch)
            if bnm_scheduler is not None:
               bnm_scheduler.step(epoch-1)
            points, target = data
            points, target = points.cuda(), target.cuda()
            points, target = Variable(points), Variable(target)
            
            # fastest point sampling
            #print(points.shape[1], args.num_points)
            fps_idx = pointnet2_utils.furthest_point_sample(points, args.num_points)  # (B, npoint)
            #fps_idx = fps_idx[:, np.random.choice(3600, args.num_points, False)]
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
            
            # augmentation
            points.data = PointcloudScaleAndTranslate(points.data)
            PointcloudRandomInputDropout = d_utils.PointcloudRandomInputDropout()
            points.data = PointcloudRandomInputDropout(points.data)
            
            optimizer.zero_grad()
            
            pred = model(points)
            target = target.view(-1)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            """## my code 2020/2/27
            errors = {'train_loss': loss.item()}
            epoch_iter += args.batch_size
            epoch_ratio = float(epoch_iter)/(num_batch * args.batch_size) # num_batch = len(train_dataset)/args.batch_size
            plot_data['X'].append(epoch+epoch_ratio)
            plot_data['Y'].append([errors[k] for k in plot_data['legend']])
            plot_current_errors(plot_data)
            ##"""
            if i % args.print_freq_iter == 0:
                print('[epoch %3d: %3d/%3d] \t train loss: %0.6f \t lr: %0.5f' %(epoch+1, i, num_batch, loss.data.clone(), lr_scheduler.get_lr()[0]))
            batch_count += 1
            
            # validation in between an epoch
            if args.evaluate and batch_count % int(args.val_freq_epoch * num_batch) == 0:
                validate(test_dataloader, model, criterion, args, batch_count)
        

def validate(test_dataloader, model, criterion, args, iter): 
    global g_acc
    model.eval()
    losses, preds, labels = [], [], []
    with torch.no_grad():
        for j, data in enumerate(test_dataloader, 0):
            points, target = data
            points, target = points.cuda(), target.cuda()
            points, target = Variable(points), Variable(target)
            
            # fastest point sampling
            fps_idx = pointnet2_utils.furthest_point_sample(points, args.num_points)  # (B, npoint)
            # fps_idx = fps_idx[:, np.random.choice(1200, args.num_points, False)]
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()

            pred = model(points)
            target = target.view(-1)
            loss = criterion(pred, target)
            losses.append(loss.data.clone())
            _, pred_choice = torch.max(pred.data, -1)

            
            preds.append(pred_choice)
            labels.append(target.data)
            
        preds = torch.cat(preds, 0)
        labels = torch.cat(labels, 0)
        #print(torch.sum(preds == labels), labels.numel())
        acc = torch.sum(preds == labels).item()/labels.numel()
        print('\nval loss: %0.6f \t acc: %0.6f\n'%(np.array(losses).mean(), acc))
        
        if acc > g_acc:
            g_acc = acc
            torch.save(model.state_dict(), '%s/face_id_ssn_iter_%d_acc_%0.6f_all.pth' % (args.save_path, iter, acc))
            # compute accurarcy of each category
            """tmp_preds = preds.cpu().numpy()
            tmp_labels = labels.cpu().numpy()
            total_cate = np.zeros(labels.numel(), dtype=np.int)
            acc_cate = np.zeros(labels.numel(), dtype=np.int)
            for i in np.arange(labels.numel()):
                total_cate[tmp_labels[i]] += 1
                if tmp_preds[i] == tmp_labels[i]:
                    acc_cate[tmp_labels[i]] +=1
            with open('log/face_expression_acc_10.txt', 'w') as m_f:
                m_f.write('\t\tSU\t'+'DI\t'+'FE\t'+'AN\t'+'SA\t'+'NE\t'+'HA\n')
                m_f.write('acc:\t')
                for i in np.arange(tmp_labels.shape[0]):  
                    m_f.write(str(int(acc_cate[i]))+'\t')
                m_f.write('\nall:\t')
                for i in np.arange(tmp_labels.shape[0]):  
                    m_f.write(str(int(total_cate[i]))+'\t')
            """

    model.train()
    
if __name__ == "__main__":
    main()