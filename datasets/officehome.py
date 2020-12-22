# -*- coding: utf-8 -*-
import os
import sys
import copy
import random
import numpy as np
import torch
from torchvision import transforms
from .datasets import register_dataset
import utils

@register_dataset('OfficeHome')
class OfficeHomeDataset:
	"""
	OfficeHome Dataset class
	"""

	def __init__(self, name, img_dir, LDS_type, is_target):
		self.name = name
		self.img_dir = img_dir
		self.LDS_type = LDS_type
		self.is_target = is_target

	def get_data(self):
		normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

		self.train_transforms = transforms.Compose([
					transforms.Resize(256),
					transforms.RandomCrop((224, 224)),
					transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
					normalize_transform
				])
		self.test_transforms = transforms.Compose([
				transforms.Resize((224, 224)),
				transforms.ToTensor(),
				normalize_transform
			])

		if self.LDS_type == 'natural':
			train_path = os.path.join('data/OfficeHome/txt/', '{}.txt'.format(self.name))
			test_path = os.path.join('data/OfficeHome/txt/', '{}.txt'.format(self.name))
		elif self.LDS_type == 'RS_UT':
			shift = 'UT' if self.is_target else 'RS'
			train_path = os.path.join('data/OfficeHome/txt/', '{}_{}.txt'.format(self.name, shift))
			test_path = os.path.join('data/OfficeHome/txt/', '{}_{}.txt'.format(self.name, shift))
		else: raise NotImplementedError

		train_dataset = utils.ImageList(open(train_path).readlines(), os.path.join(self.img_dir, 'images'))
		val_dataset = utils.ImageList(open(test_path).readlines(), os.path.join(self.img_dir, 'images'))
		test_dataset = utils.ImageList(open(test_path).readlines(), os.path.join(self.img_dir, 'images'))

		self.num_classes = 65
		train_dataset.targets, val_dataset.targets, test_dataset.targets = torch.from_numpy(train_dataset.labels), \
																		   torch.from_numpy(val_dataset.labels), \
																		   torch.from_numpy(test_dataset.labels)
		return self.num_classes, train_dataset, val_dataset, test_dataset, self.train_transforms, self.test_transforms
