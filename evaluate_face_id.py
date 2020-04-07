import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
from torchvision import transforms
from models import RSCNN_SSN_Cls as RSCNN_SSN
from data import Bosphorus_eval
import utils.pytorch_utils as pt_utils
import utils.pointnet2_utils as pointnet2_utils
import data.data_utils as d_utils
import argparse
import random
import yaml

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)            
torch.cuda.manual_seed(seed)       
torch.cuda.manual_seed_all(seed) 

parser = argparse.ArgumentParser(description='Relation-Shape CNN Shape Classification Voting Evaluation')
parser.add_argument('--config', default='cfgs/config_ssn_face_id.yaml', type=str)

NUM_REPEAT = 10
#NUM_VOTE = 10

def main():
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    for k, v in config['common'].items():
        setattr(args, k, v)
    
    test_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])

    test_dataset = Bosphorus_eval(num_points = args.num_points, root = args.data_root, transforms=test_transforms)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=int(args.workers)
    )
    
    model = RSCNN_SSN(num_classes = args.num_classes, input_channels = args.input_channels, relation_prior = args.relation_prior, use_xyz = True)
    model.cuda()
    
    if args.checkpoint is not '':
        model.load_state_dict(torch.load(args.checkpoint))
        print('Load model successfully: %s' % (args.checkpoint))
    
    # model is used for feature extraction, so no need FC layers
    model.FC_layer = nn.Linear(1024, 1024, bias=False).cuda()
    for para in model.parameters():
        para.requires_grad = False
    nn.init.eye_(model.FC_layer.weight)

    # evaluate
    #PointcloudScale = d_utils.PointcloudScale()   # initialize random scaling
    model.eval()
    global_acc = 0
    with torch.no_grad():
        Total_samples = 0
        Correct = 0
        gallery_points, gallery_labels = test_dataset.get_gallery()
        gallery_points, gallery_labels = gallery_points.cuda(), gallery_labels.cuda()
        gallery_points =  Variable(gallery_points)
        gallery_pred = model(gallery_points)
        print(gallery_pred.size())
        gallery_pred = F.normalize(gallery_pred)

        for j, data in enumerate(test_dataloader, 0):
            probe_points, probe_labels = data
            probe_points, probe_labels = probe_points.cuda(), probe_labels.cuda()           
            probe_points = Variable(probe_points)
            
            # get feature vetor for probe and gallery set from model
            probe_pred = model(probe_points)           
            probe_pred = F.normalize(probe_pred)
           
            # make tensor to size (probe_num, gallery_num, C)
            probe_tmp = probe_pred.unsqueeze(1).expand(probe_pred.shape[0], gallery_pred.shape[0], 
                                                        probe_pred.shape[1])
            gallery_tmp = gallery_pred.unsqueeze(0).expand(probe_pred.shape[0], gallery_pred.shape[0],
                                                        gallery_pred.shape[1])
            results = torch.sum(torch.mul(probe_tmp, gallery_tmp), dim=2) # cosine distance
            results = torch.argmax(results, dim=1)
            
            Total_samples += probe_points.shape[0]
            for i in np.arange(0, results.shape[0]):
                if gallery_labels[results[i]] == probe_labels[i]:
                    Correct += 1
        print('Total_samples:{}'.format(Total_samples))      
        acc = float(Correct/Total_samples)
        if acc > global_acc:
            global_acc = acc
        print('Repeat %3d \t Acc: %0.6f' % (i + 1, acc))
    print('\nBest voting acc: %0.6f' % (global_acc))
        
if __name__ == '__main__':
    main()