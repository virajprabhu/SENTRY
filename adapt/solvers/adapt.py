# -*- coding: utf-8 -*-
import sys
from tqdm import tqdm
import random
import copy

import numpy as np
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F

from .solver import register_solver
sys.path.append('../../')
import utils

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)
torch.cuda.manual_seed(1234)

class BaseSolver:
	def __init__(self, net, src_loader, tgt_loader, train_idx, tgt_opt, device, num_classes, args):
		self.net = net
		self.src_loader = src_loader
		self.tgt_loader = tgt_loader
		self.train_idx = np.array(train_idx)
		self.tgt_opt = tgt_opt
		self.device = device	
		self.num_classes = num_classes
		self.args = args
		self.current_step = 0
		self.param_lr_c = []
		for param_group in self.tgt_opt.param_groups:
			self.param_lr_c.append(param_group["lr"])

	def lr_step(self):
		"""
		Learning rate scheduler
		"""
		if self.args.optimizer == 'SGD':
			self.tgt_opt = utils.inv_lr_scheduler(self.param_lr_c, self.tgt_opt, self.current_step, init_lr=self.args.lr)

	def solve(self, epoch):
		pass

@register_solver('dann')
class DANNSolver(BaseSolver):
	"""
	Implements DANN from Unsupervised Domain Adaptation by Backpropagation: https://arxiv.org/abs/1409.7495
	"""
	def __init__(self, net, src_loader, tgt_loader, train_idx, tgt_opt, device, num_classes, args):
		super(DANNSolver, self).__init__(net, src_loader, tgt_loader, train_idx, tgt_opt, device, num_classes, args)
	
	def solve(self, epoch, disc, disc_opt):
		"""
		Unsupervised adaptation via DANN: https://arxiv.org/abs/1505.07818
		XE on labeled source + Domain Adversarial training on source+target
		"""
		gan_criterion = nn.CrossEntropyLoss()

		self.net.train()
		disc.train()
		
		lambda_src, lambda_unsup = self.args.lambda_src, self.args.lambda_unsup

		joint_loader = zip(self.src_loader, self.tgt_loader)	
		
		for batch_idx, (((data_s, _, _), label_s, _), ((data_t, _, _), _, _)) in enumerate(tqdm(joint_loader)):
			self.current_step += 1
			self.tgt_opt.zero_grad()
			disc_opt.zero_grad()

			data_s, label_s = data_s.to(self.device), label_s.to(self.device)
			data_t = data_t.to(self.device)

			# Train with target labels
			score_s = self.net(data_s)
			xeloss_src = lambda_src*nn.CrossEntropyLoss()(score_s, label_s)
			
			info_str = "[Train DANN] Epoch: {}".format(epoch)
			info_str += " Src Sup loss: {:.3f}".format(xeloss_src.item())                    

			# extract and concat features
			score_t = self.net(data_t)
			f = torch.cat((score_s, score_t), 0)

			# predict with discriminator
			f_rev = utils.ReverseLayerF.apply(f)
			pred_concat = disc(f_rev)

			target_dom_s = torch.ones(len(data_s)).long().to(self.device)
			target_dom_t = torch.zeros(len(data_t)).long().to(self.device)
			label_concat = torch.cat((target_dom_s, target_dom_t), 0)

			# compute loss for disciminator
			loss_domain = gan_criterion(pred_concat, label_concat)

			loss_final = (xeloss_src) + (lambda_unsup * loss_domain)
			loss_final.backward()

			self.tgt_opt.step()
			disc_opt.step()

			# Learning rate update (if using SGD)
			self.lr_step()

			# log net update info
			info_str += " DANN loss: {:.3f}".format(lambda_unsup * loss_domain.item())		
		
			if batch_idx%10 == 0: print(info_str)

@register_solver('SENTRY')
class SENTRYSolver(BaseSolver):
	"""
	Implements SENTRY
	"""
	def __init__(self, net, src_loader, tgt_loader, train_idx, tgt_opt, device, num_classes, args):
		super(SENTRYSolver, self).__init__(net, src_loader, tgt_loader, train_idx, tgt_opt, device, num_classes, args)
		self.num_classes = args.num_classes
		self.queue_length = 256 	# Queue length for computing target information entropy loss
		
		# Committee consistency hyperparameters
		self.randaug_n = 3			# RandAugment number of consecutive transformations
		self.randaug_m = 2.0		# RandAugment severity
		self.committee_size = 3 	# Committee size		
		self.positive_threshold, self.negative_threshold = (self.committee_size // 2) + 1, \
														   (self.committee_size // 2) + 1  # Use majority voting scheme		 
		# Pass in hyperparams to dataset
		self.tgt_loader.dataset.committee_size = self.committee_size
		self.tgt_loader.dataset.ra_obj.n = self.randaug_n
		self.tgt_loader.dataset.ra_obj.m = self.randaug_m
		
	def compute_prf1(self, true_mask, pred_mask):
		"""
		Compute precision, recall, and F1 metrics for predicted mask against ground truth
		"""
		conf_mat = confusion_matrix(true_mask, pred_mask, labels=[False, True])
		p = conf_mat[1, 1] / (conf_mat[0, 1] + conf_mat[1, 1] + 1e-8)
		r = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1] + 1e-8)
		f1 = (2*p*r) / (p+r+1e-8)
		return conf_mat, p, r, f1

	def solve(self, epoch):
		"""
		Unsupervised Domain adaptation via SENTRY: Selective Entropy Optimization via Committee Consistency
		"""
		
		joint_loader = zip(self.src_loader, self.tgt_loader)
		
		lambda_src, lambda_unsup, lambda_ent = self.args.lambda_src, self.args.lambda_unsup, self.args.lambda_ent
		
		self.net.train()
		queue = torch.zeros(self.queue_length).to(self.device)
		pointer = 0
		for batch_idx, (((data_s, _, _), label_s, _), ((_, data_t_og, data_t_raug), label_t, indices_t)) in enumerate(tqdm(joint_loader)):
			self.current_step += 1
			data_s, label_s = data_s.to(self.device), label_s.to(self.device)
			data_t_og, label_t = data_t_og.to(self.device), label_t.to(self.device)
			
			# Train with target labels
			score_s = self.net(data_s)
			xeloss_src = lambda_src*nn.CrossEntropyLoss()(score_s, label_s)
			loss = xeloss_src

			info_str = "\n[Train SENTRY] Epoch: {}".format(epoch)
			info_str += " Source XE loss: {:.3f}".format(xeloss_src.item())                    

			score_t_og = self.net(data_t_og)
			batch_sz = data_t_og.shape[0]
			tgt_preds = score_t_og.max(dim=1)[1].reshape(-1)

			if pointer+batch_sz > self.queue_length: # Deal with wrap around when ql % batchsize != 0 
				rem_space = self.queue_length-pointer
				queue[pointer: self.queue_length] = (tgt_preds[:rem_space].detach()+1)
				queue[0:batch_sz-rem_space] = (tgt_preds[rem_space:].detach()+1)
			else: 
				queue[pointer: pointer+batch_sz] = (tgt_preds.detach()+1)
			pointer = (pointer+batch_sz) % self.queue_length

			bincounts = torch.bincount(queue.long(), minlength=self.num_classes+1).float() / self.queue_length
			bincounts = bincounts[1:]
			
			log_q = torch.log(bincounts + 1e-12).detach()
			loss_infoent = lambda_unsup * torch.mean(torch.sum(score_t_og.softmax(dim=1) * log_q.reshape(1, self.num_classes), dim=1))
			loss += loss_infoent
			info_str += " Infoent loss: {:.3f}".format(loss_infoent.item())

			score_t_og = self.net(data_t_og).detach()
			tgt_preds = score_t_og.max(dim=1)[1].reshape(-1)
			
			# When pseudobalancing, label_t will correspond to pseudolabels rather than ground truth, so use backup instead
			if self.args.pseudo_balance_target: label_t = self.tgt_loader.dataset.targets_copy[indices_t]
			
			# Compute actual correctness mask for analysis only
			correct_mask_gt = (tgt_preds.detach().cpu() == label_t)

			correct_mask, incorrect_mask = torch.zeros_like(tgt_preds).to(self.device), \
											torch.zeros_like(tgt_preds).to(self.device)					

			score_t_aug_pos, score_t_aug_neg = torch.zeros_like(score_t_og), torch.zeros_like(score_t_og)
			for data_t_aug_curr in data_t_raug:
				score_t_aug_curr = self.net(data_t_aug_curr.to(self.device))
				tgt_preds_aug = score_t_aug_curr.max(dim=1)[1].reshape(-1)
				consistent_idxs = (tgt_preds == tgt_preds_aug).detach()
				inconsistent_idxs = (tgt_preds != tgt_preds_aug).detach()
				correct_mask = correct_mask + consistent_idxs.type(torch.uint8)						
				incorrect_mask = incorrect_mask + inconsistent_idxs.type(torch.uint8)

				score_t_aug_pos[consistent_idxs, :] = score_t_aug_curr[consistent_idxs, :]
				score_t_aug_neg[inconsistent_idxs, :] = score_t_aug_curr[inconsistent_idxs, :]
			
			correct_mask, incorrect_mask = correct_mask>=self.positive_threshold, incorrect_mask>=self.negative_threshold
			
			# Compute some stats
			correct_ratio = (correct_mask).sum().item() / data_t_og.shape[0]				
			incorrect_ratio = (incorrect_mask).sum().item() / data_t_og.shape[0]
			consistency_conf_mat, correct_precision, correct_recall, correct_f1 = self.compute_prf1(correct_mask_gt.cpu().numpy(), \
																									correct_mask.cpu().numpy())
			info_str += "\n {:d} / {:d} consistent ({:.2f}): GT precision: {:.2f}".format(correct_mask.sum(), data_t_og.shape[0], \
																						  correct_ratio, correct_precision)
			
			if correct_ratio > 0.0:
				probs_t_pos = F.softmax(score_t_aug_pos, dim=1)		
				loss_cent_correct = lambda_ent * correct_ratio * -torch.mean(torch.sum(probs_t_pos[correct_mask] * \
																			(torch.log(probs_t_pos[correct_mask] + 1e-12)), 1))
				loss += loss_cent_correct
				info_str += " SENTRY loss (consistent): {:.3f}".format(loss_cent_correct.item())
			
			if incorrect_ratio > 0.0:
				probs_t_neg = F.softmax(score_t_aug_neg, dim=1)
				loss_cent_incorrect = lambda_ent * incorrect_ratio * torch.mean(torch.sum(probs_t_neg[incorrect_mask] * \
																				(torch.log(probs_t_neg[incorrect_mask] + 1e-12)), 1))
				loss += loss_cent_incorrect
				info_str += " SENTRY loss (inconsistent): {:.3f}".format(loss_cent_incorrect.item())

			# Backprop
			self.tgt_opt.zero_grad()
			loss.backward()
			self.tgt_opt.step()

			# Learning rate update (if using SGD)
			self.lr_step()

			if batch_idx%10 == 0: print(info_str)