# -*- coding: utf-8 -*-
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.utils.data as data
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
index = str(BASE_DIR).index('data')
if not BASE_DIR[0:index] in sys.path:
  sys.path.append(BASE_DIR[0:index])

import utils.pointnet2_utils as pointnet2_utils


def pc_normalize(pc):
	# pc.shape[0] is the number of points
	# pc.shape[1] is the number of feature dimensions
	centroid = np.mean(pc, axis=0)
	pc = pc - centroid
	dist = np.max(np.sqrt(np.sum(pc**2, axis=1)))
	pc = pc / dist
	return pc

def _get_data_files(list_filename):
	with open(list_filename) as f:
		content = f.readlines()
		filenames = [line.strip().split('\t')[0] for line in content]
		labels = [int(line.strip().split('\t')[1]) for line in content]
		return  filenames, labels

class BU3DFE(data.Dataset):
	def __init__(self, num_points, root, transforms=None, train=True, task='expression'):
		super().__init__()

		self.transforms = transforms
		self.num_points = num_points
		self.root = os.path.abspath(root)
		self.folder = 'Only_pts_BU3DFE'
		self.data_dir = os.path.join(self.root, self.folder)

		self.train = train
		if task=='expression':
			if self.train:
				self.files, self.labels = _get_data_files(
											os.path.join(self.data_dir, 'train_expression_6.txt')
											)
			else:
				self.files, self.labels = _get_data_files(
											os.path.join(self.data_dir, 'test_expression_6.txt')
											)
		elif task=='id':
			if self.train:
				self.files, self.labels = _get_data_files(
											os.path.join(self.data_dir, 'BU3DFE_id_all.txt')
											)
			else:
				self.files, self.labels = _get_data_files(
											os.path.join(self.data_dir, 'train_id.txt')
											)
		#print(self.files[0:10], self.labels[0:10])
		self.points = []
		np.random.seed(19970513)
		for file in self.files:
			single_p = np.loadtxt(os.path.join(self.root, file))
			single_p = pc_normalize(single_p)
			if self.num_points > single_p.shape[0]:
				idx = np.ones(single_p.shape[0], dtype=np.int32)
				idx[-1] = self.num_points - single_p.shape[0] + 1
				single_p_part = np.repeat(single_p, idx, axis=0)
			else:
				#idxs = np.random.choice(single_p.shape[0], self.num_points, replace=False)
				#single_p_part = single_p[idxs].copy()
				single_p_tensor = torch.from_numpy(single_p).type(torch.FloatTensor).cuda()
				single_p_tensor = single_p_tensor.unsqueeze(0) # change to (1, N, 3)
				fps_idx = pointnet2_utils.furthest_point_sample(single_p_tensor,self.num_points) # (1, npoint)
				single_p_tensor = pointnet2_utils.gather_operation(single_p_tensor.transpose(1,2).contiguous(), 
																	fps_idx).transpose(1,2).contiguous() # (1, npoint, 3)
				single_p_part = single_p_tensor.squeeze(0).cpu().numpy()
			#print(single_p.shape)
			self.points.append(single_p_part)

		self.points = np.stack(self.points, axis=0) # need to be tested
		#self.points = np.array(self.points)
		self.labels = np.array(self.labels)

	def __getitem__(self, idx):
		pt_idxs = np.arange(0, self.points.shape[1])
		if self.train:
			np.random.shuffle(pt_idxs)

		current_points = self.points[idx, pt_idxs].copy()
		if self.transforms is not None:
			current_points = self.transforms(current_points)
		else:
			current_points = torch.from_numpy(current_points).float()

		label = torch.Tensor([self.labels[idx]]).type(torch.LongTensor)

		return current_points, label

	def __len__(self):
		return self.points.shape[0]

class BU3DFE_eval(data.Dataset):
	def __init__(self, num_points, root, transforms=None, task='id'):
		super().__init__()

		self.transforms = transforms
		self.num_points = num_points
		self.root = os.path.abspath(root)
		self.folder = 'Only_pts_BU3DFE'
		self.data_dir = os.path.join(self.root, self.folder)

		if task=='id' :
			self.probe_files, self.probe_labels = _get_data_files(
										os.path.join(self.data_dir, 'probe.txt')
										)
			
			self.gallery_files, self.gallery_labels = _get_data_files(
										os.path.join(self.data_dir, 'gallery.txt')
										)

		self.probe_points = []
		self.gallery_points = []
		for file in self.probe_files:
			single_p = np.loadtxt(os.path.join(self.root, file), delimiter=',')
			single_p = pc_normalize(single_p)
			if self.num_points > single_p.shape[0]:
				idx = np.ones(single_p.shape[0], dtype=np.int32)
				idx[-1] = self.num_points - single_p.shape[0] + 1
				single_p_part = np.repeat(single_p, idx, axis=0)
			else:
				#idxs = np.random.choice(single_p.shape[0], self.num_points, replace=False)
				#single_p_part = single_p[idxs].copy()
				single_p_tensor = torch.from_numpy(single_p).type(torch.FloatTensor).cuda()
				single_p_tensor = single_p_tensor.unsqueeze(0) # change to (1, N, 3)
				fps_idx = pointnet2_utils.furthest_point_sample(single_p_tensor,self.num_points) # (1, npoint)
				single_p_tensor = pointnet2_utils.gather_operation(single_p_tensor.transpose(1,2).contiguous(), 
																	fps_idx).transpose(1,2).contiguous() # (1, npoint, 3)
				single_p_part = single_p_tensor.squeeze(0).cpu().numpy()

			#print(single_p.shape)
			self.probe_points.append(single_p_part)

		for file in self.gallery_files:
			single_p = np.loadtxt(os.path.join(self.root, file), delimiter=',')
			single_p = pc_normalize(single_p)
			if self.num_points > single_p.shape[0]:
				idx = np.ones(single_p.shape[0], dtype=np.int32)
				idx[-1] = self.num_points - single_p.shape[0] + 1
				single_p_part = np.repeat(single_p, idx, axis=0)
			else:
				single_p_tensor = torch.from_numpy(single_p).type(torch.FloatTensor).cuda()
				single_p_tensor = single_p_tensor.unsqueeze(0) # change to (1, N, 3)
				fps_idx = pointnet2_utils.furthest_point_sample(single_p_tensor,self.num_points) # (1, npoint)
				single_p_tensor = pointnet2_utils.gather_operation(single_p_tensor.transpose(1,2).contiguous(), 
																	fps_idx).transpose(1,2).contiguous() # (1, npoint, 3)
				single_p_part = single_p_tensor.squeeze(0).cpu().numpy()

			#print(single_p.shape)
			self.gallery_points.append(single_p_part)		

		self.probe_points = np.stack(self.probe_points, axis=0)
		self.gallery_points = np.stack(self.gallery_points, axis=0)  # need to be tested
		
		self.probe_labels = np.array(self.probe_labels)
		self.gallery_labels = np.array(self.gallery_labels)


	def __getitem__(self, idx):
		#pt_idxs = np.arange(0, self.points.shape[1])
		
		#np.random.shuffle(pt_idxs)

		current_points = self.probe_points[idx, :].copy()
		if self.transforms is not None:
			current_points = self.transforms(current_points)
		else:
			current_points = torch.from_numpy(current_points).float()

		probe_label = torch.Tensor([self.probe_labels[idx]]).type(torch.LongTensor)

		return current_points, probe_label

	def __len__(self):
		return self.probe_points.shape[0]

	def get_gallery(self):
		if self.transforms is not None:
			g_points = self.transforms(self.gallery_points)
		else:
			g_points = torch.from_numpy(self.gallery_points).float()

		g_labels = torch.from_numpy(self.gallery_labels).type(torch.LongTensor)
		return g_points, g_labels

if __name__ == '__main__':
	dset = BU3DFE(4000, "/home/cv_jcy/Pytorch_Workspace/pointnet.pytorch/3d_dataset/", train=True)
	print(len(dset), dset.points.shape)
	print(dset[99][0])
	print(dset[99][1])
	print(len(dset))