import os
import copy
import random
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.autograd import Function

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

######################################################################
##### Data loading utilities
######################################################################

def default_loader(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		with Image.open(f) as img:
			return img.convert('RGB')

def make_dataset(image_list, labels=None):
	if labels:
		len_ = len(image_list)
		images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
	else:
		if len(image_list[0].split()) > 2:
			images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
		else:
			images = [(val.split()[0], int(val.split()[1])) for val in image_list]
	return images

class ImageList(object):
	"""A generic data loader where the images are arranged in this way: ::
		root/dog/xxx.png
		root/dog/xxy.png
		root/dog/xxz.png
		root/cat/123.png
		root/cat/nsdf3.png
		root/cat/asd932_.png
	Args:
		root (string): Root directory path.
		transform (callable, optional): A function/transform that  takes in an PIL image
			and returns a transformed version. E.g, ``transforms.RandomCrop``
		target_transform (callable, optional): A function/transform that takes in the
			target and transforms it.
		loader (callable, optional): A function to load an image given its path.
	 Attributes:
		classes (list): List of the class names.
		class_to_idx (dict): Dict with items (class_name, class_index).
		imgs (list): List of (image path, class_index) tuples
	"""

	def __init__(self, image_list, root, transform=None, target_transform=None,
				 loader=default_loader):
		imgs = make_dataset(image_list)
		if len(imgs) == 0:
			raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"))

		self.root = root
		self.data = np.array([os.path.join(self.root, img[0]) for img in imgs])
		self.labels = np.array([img[1] for img in imgs])
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index
		Returns:
			tuple: (image, target) where target is class_index of the target class.
		"""
		path, target = self.data[index], self.labels[index]
		path = os.path.join(self.root, path)
		img = self.loader(path)
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target, index

	def __len__(self):
		return len(self.data)

######################################################################
##### Custom layers
######################################################################

class ReverseLayerF(Function):

	@staticmethod
	def forward(ctx, x):
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		output = grad_output.neg()
		return output, None

######################################################################
##### Optimization utilities
######################################################################

def generate_optimizer(model, args, mode='da'):
	optimizers = {}
	param_list = []
	lr = args.lr if mode == 'source' else args.da_lr
	wd = args.wd if mode == 'source' else args.da_wd
	if args.optimizer == 'Adam':
		optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999))
	elif args.optimizer == 'SGD':
		for key, value in dict(model.named_parameters()).items():
			if value.requires_grad:
				if 'classifier' not in key:
					param_list += [{'params': [value], 'lr': lr*0.1, 'weight_decay': wd}]
				else:
					param_list += [{'params': [value], 'lr': lr, 'weight_decay': wd}]

		optimizer = optim.SGD(param_list, momentum=0.9, weight_decay=wd, nesterov=True)
	else: raise NotImplementedError

	return optimizer

def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma=0.0001, power=0.75, init_lr=0.001):
	"""
	Decay learning rate
	"""	
	for i, param_group in enumerate(optimizer.param_groups):
		lr = param_lr[i] * (1 + gamma * iter_num) ** (- power)
		param_group['lr'] = lr
	return optimizer

######################################################################
##### Training utilities
######################################################################

def train_source_model(source_model, src_train_loader, src_val_loader, num_classes, args, device):
	best_acc, best_source_model = 0.0, None
	source_model.to(device)

	params = []
	source_optimizer = generate_optimizer(source_model, args, mode='source')
	source_file = '{}_{}_source.pth'.format(args.source, args.cnn)
	for epoch in tqdm(range(args.num_epochs)):
		train_acc, train_loss = train(source_model, device, src_train_loader, source_optimizer, epoch)
		out_str = 'Epoch {}:  Train Loss={:.3f}\tTrain Acc={:.2f}'.format(epoch, train_loss, train_acc)
		if args.source in ['mnist', 'svhn']: # Other benchmarks don't use a validation set
			val_acc, conf_mat = test(source_model, device, src_val_loader, split='val', num_classes=num_classes)
			val_avg_acc = (conf_mat.diagonal().numpy() / conf_mat.sum(axis=1).numpy()).mean() * 100
			out_str += '\tVal Avg. Acc.={:.2f} '.format(val_avg_acc)
			if (val_avg_acc > best_acc):
				best_acc = val_avg_acc
				best_source_model = copy.deepcopy(source_model)
		else:
			best_source_model = source_model
		
		print(out_str)

	print('Saving to', os.path.join('checkpoints', 'source', source_file))
	torch.save(best_source_model.state_dict(), os.path.join('checkpoints', 'source', source_file))
	return best_source_model

def train(model, device, train_loader, optim, epoch):
	model.train()
	total_loss, avg_loss = 0.0, 0.0
	correct, total_el = 0.0, 0.0
	for batch_idx, ((data, _, _), target, _) in enumerate(tqdm(train_loader)):
		data, target = data.to(device), target.to(device)
		optim.zero_grad()
		output = model(data)
		loss = nn.CrossEntropyLoss()(output, target)
		pred = output.argmax(dim=1, keepdim=True)
		correct += pred.eq(target.view_as(pred)).sum().item()
		total_loss += loss.item()
		loss.backward()
		optim.step()
		total_el += data.shape[0]

	train_acc = 100. * correct / total_el
	train_loss = total_loss / len(train_loader.sampler)
	return train_acc, train_loss

def test(model, device, test_loader, split="test", num_classes=10):
	model.eval()
	correct = 0
	confusion_matrix = torch.zeros(num_classes, num_classes).long()
	with torch.no_grad():
		for data, target, _ in tqdm(test_loader):
			data = data[0]
			data, target = data.to(device), target.to(device)
			output = model(data)
			pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
			scores = nn.Softmax(dim=1)(output)
			correct += pred.eq(target.view_as(pred)).sum().item()
			for t, p in zip(target.view(-1), pred.view(-1)):
				confusion_matrix[t.long(), p.long()] += 1
	test_acc = 100. * correct / len(test_loader.sampler)
	return test_acc, confusion_matrix

def get_embedding(model, loader, device, num_classes, args, emb_dim=512):
	model.eval()

	embedding = torch.zeros([len(loader.dataset), num_classes])
	labels = torch.zeros(len(loader.dataset)).long()
	preds = torch.zeros(len(loader.dataset)).long()

	with torch.no_grad():
		for batch_idx, ((data, _, _), target, indices) in enumerate(tqdm(loader)):
			data = data.to(device)
			e1 = model(data)
			embedding[indices, :] = e1.cpu()
			labels[indices] = target
			preds[indices] = e1.argmax(dim=1, keepdim=True).squeeze().cpu()
	
	nz_idxs = torch.arange(len(loader.dataset))[embedding.sum(dim=1) != 0]
	embedding = embedding[nz_idxs]
	labels = labels[nz_idxs]
	preds = preds[nz_idxs]
	
	return embedding, labels, preds

######################################################################
##### Plotting utilities
######################################################################

def plot_cm(ax, conf_matrix, args, label='Confusion matrix', num_classes=10):
	df_cm = pd.DataFrame(conf_matrix.cpu().numpy(), \
						 index = [i for i in range(num_classes)], \
						 columns = [i for i in range(num_classes)])
	if num_classes > 10:
		sns.heatmap(df_cm, annot=False, fmt="d", ax=ax, cbar=False)
	else:
		sns.heatmap(df_cm, annot=True, fmt="d", ax=ax, cbar=False)

	ax.set_title(label, fontsize=22)
	ax.set_xlabel('Prediction', fontsize=22)
	ax.set_ylabel('Ground Truth', fontsize=22)

def plot_per_class_diff(ax, per_class_after, per_class_before, num_classes, train_loader, title):
	tgts = train_loader.dataset.targets
	if isinstance(tgts, torch.Tensor): tgts = tgts.numpy() 
	counts = Counter(tgts)

	sorted_counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True)}
	order = np.array(list(sorted_counts.keys()))
	X = np.array([ix for ix in range(num_classes)])

	ax.bar(X, per_class_after[order] - np.array(per_class_before)[order], color="#4389E5", alpha=0.8)
	ax.set_xticks(X)
	ax.set_xticklabels(order, fontsize=14, rotation=90)
	ax.set_ylim(-1.0, 1.0)
	ax.set_yticklabels([np.round(ytick*100, 2) for ytick in ax.get_yticks()], fontsize=18)
	ax.set_xlabel(r'Ground truth label (size decreases $\rightarrow$)', fontsize=22)
	ax.set_ylabel('Accuracy change (%)', fontsize=22)
	ax.set_xlim(-0.5, num_classes-0.5)
	ax.set_title(title, fontsize=22)

def plot_accuracy_statistics(cm_before, cm_after, num_classes, args, target_train_loader):
	pca_before = cm_before.diagonal().numpy() / cm_before.sum(axis=1).numpy()
	pca_after = cm_after.diagonal().numpy() / cm_after.sum(axis=1).numpy()
	
	matplotlib.rcParams.update({'font.size': 16})	
	fig, axs = plt.subplots(1, 3, figsize=(22, 7))

	plot_cm(axs[0], cm_before, args, label='Before {} (Avg. acc: {:.2f})'.format(args.da_strat, pca_before.mean()*100), num_classes=num_classes)
	plot_cm(axs[1], cm_after, args, label='After {} (Avg. acc: {:.2f})'.format(args.da_strat, pca_after.mean()*100), num_classes=num_classes)
	plot_per_class_diff(axs[2], pca_after, pca_before, num_classes, \
						target_train_loader, title='SENTRY: Per-class accuracy change')

	fig.suptitle(r'{}$\rightarrow${}-LT ({}): {}'.format(args.source, args.target, args.LDS_type, args.da_strat), fontsize=28, y=1.05)
	plt.tight_layout()
	plt.savefig('results/{}_{}_{}_{}_acc_analysis.pdf'.format(args.source, args.target, args.LDS_type, args.da_strat), \
															  bbox_inches='tight', dpi=300)