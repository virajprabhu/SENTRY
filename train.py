# Domain Adaptation experiments
import os
import random
import argparse
import copy
import pprint
import distutils
import distutils.util
from omegaconf import OmegaConf

import numpy as np
from tqdm import tqdm

import torch

from adapt.models.models import get_model
from adapt.solvers.solver import get_solver
from datasets.base import UDADataset
import utils
from adapt import *

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)
torch.cuda.manual_seed(1234)

def main():	
	parser = argparse.ArgumentParser()
	# Load existing configuration?
	parser.add_argument('--load_from_cfg', type=lambda x:bool(distutils.util.strtobool(x)), default=False, help="Load from config?")
	parser.add_argument('--cfg_file', type=str, help="Experiment configuration file", default="config/digits/dann.yml")
	# Experiment identifer
	parser.add_argument('--id', type=str, help="Experiment identifier")
	parser.add_argument('--use_cuda', help="Use GPU?")
	# Source and target domain
	parser.add_argument('--source', help="Source dataset")
	parser.add_argument('--target', help="Target dataset")
	parser.add_argument('--img_dir', type=str, default="data/", help="Data directory where images are stored")
	parser.add_argument('--LDS_type', type=str, default="natural", help="Label Distribution Shift type")	
	# CNN parameters
	parser.add_argument('--cnn', type=str, help="CNN architecture")	
	parser.add_argument('--load_source', type=lambda x:bool(distutils.util.strtobool(x)), default=True, help="Load source checkpoint?")	
	parser.add_argument('--l2_normalize', type=lambda x:bool(distutils.util.strtobool(x)), help="L2 normalize features?")
	parser.add_argument('--temperature', type=float, help="CNN softmax temperature")	
	# Class balancing parameters
	parser.add_argument('--class_balance_source', type=lambda x:bool(distutils.util.strtobool(x)), help="Class-balance source?")
	parser.add_argument('--pseudo_balance_target', type=lambda x:bool(distutils.util.strtobool(x)), help="Pseudo class-balance target?")
	# DA details
	parser.add_argument('--da_strat', type=str, help="DA strategy")	
	parser.add_argument('--load_da', type=lambda x:bool(distutils.util.strtobool(x)), help="Load saved DA checkpoint?")	
	# Training details		
	parser.add_argument('--optimizer', type=str, help="Optimizer")
	parser.add_argument('--batch_size', type=int, help="Batch size")	
	parser.add_argument('--lr', type=float, help="Learning rate")
	parser.add_argument('--wd', type=float, help="Weight decay")
	parser.add_argument('--num_epochs', type=int, help="Number of Epochs")
	parser.add_argument('--da_lr', type=float, help="Unsupervised DA Learning rate")
	parser.add_argument('--da_num_epochs', type=int, help="DA Number of epochs")		
	# Loss weights
	parser.add_argument('--src_sup_wt', type=float, help="Source supervised XE loss weight")
	parser.add_argument('--tgt_sup_wt', type=float, help="Target self-training XE loss weight")
	parser.add_argument('--unsup_wt', type=float, help="Target unsupervised loss weight")
	parser.add_argument('--cent_wt', type=float, help="Target entropy minimization loss weight")
	
	args_cmd = parser.parse_args()

	if args_cmd.load_from_cfg:
		args_cfg = dict(OmegaConf.load(args_cmd.cfg_file))
		args_cmd = vars(args_cmd)
		for k in args_cmd.keys():
			if args_cmd[k] is not None: args_cfg[k] = args_cmd[k]
		args = OmegaConf.create(args_cfg)
	else: 
		args = args_cmd

	pp = pprint.PrettyPrinter(indent=4)
	pp.pprint(args)
	
	device = torch.device("cuda") if args.use_cuda else torch.device("cpu")

	################################################################################################################
	#### Setup source data loaders
	################################################################################################################	
	print('Loading {} dataset'.format(args.source))
	src_dset = UDADataset(args.source, args.LDS_type, is_target=False, img_dir=args.img_dir, batch_size=args.batch_size)
	src_train_dset, _, _ = src_dset.get_dsets()

	src_train_loader, src_val_loader, src_test_loader, src_train_idx = src_dset.get_loaders(class_balance_train=args.class_balance_source)
	num_classes = src_dset.get_num_classes()
	args.num_classes = num_classes
	print('Number of classes: {}'.format(num_classes))

	################################################################################################################
	#### Train / load a source model	 
	################################################################################################################	
	
	source_model = get_model(args.cnn, num_cls=num_classes, l2_normalize=args.l2_normalize, temperature=args.temperature)
	
	source_file = '{}_{}_source.pth'.format(args.source, args.cnn)
	source_path = os.path.join('checkpoints', 'source', source_file)

	if args.load_source and os.path.exists(source_path):
		print('\nFound source checkpoint at {}'.format(source_path))
		source_model.load_state_dict(torch.load(source_path, map_location=device))
		best_source_model = source_model
	else: 		
		print('\nSource checkpoint not found, training...')
		best_source_model = utils.train_source_model(source_model, src_train_loader, src_val_loader, num_classes, args, device)

	print('Evaluating source checkpoint on {} test set...'.format(args.source))
	_, cm_source = utils.test(best_source_model, device, src_test_loader, split="test", num_classes=num_classes)
	per_class_acc_source = cm_source.diagonal().numpy() / cm_source.sum(axis=1).numpy()
	per_class_acc_source = per_class_acc_source.mean() * 100
	out_str = '{} Avg. acc.: {:.2f}% '.format(args.source, per_class_acc_source)				
	print(out_str)
	
	model = copy.deepcopy(best_source_model)

	################################################################################################################
	#### Setup target data loaders
	################################################################################################################	
	print('\nLoading {} dataset'.format(args.target))

	target_dset = UDADataset(args.target, args.LDS_type, is_target=True, img_dir=args.img_dir, valid_ratio=0, batch_size=args.batch_size)
	target_dset.get_dsets()
	
	# Manually long tail target training set for SVHN->MNIST-LT adaptation
	if args.LDS_type in ['IF1', 'IF20', 'IF50', 'IF100']:
		target_dset.long_tail_train('{}_ixs_{}'.format(args.target, args.LDS_type))
	
	print('Evaluating source checkpoint on {} test set...'.format(args.target))
	target_train_loader, target_val_loader, target_test_loader, tgt_train_idx = target_dset.get_loaders()
	acc_before, cm_before = utils.test(model, device, target_test_loader, split="test", num_classes=num_classes)
	per_class_acc_before = cm_before.diagonal().numpy() / cm_before.sum(axis=1).numpy()
	per_class_acc_before = per_class_acc_before.mean() * 100

	out_str = '{}->{}-LT ({}), Before {}:\t Avg. acc={:.2f}%\tAgg. acc={:.2f}%'.format(args.source, args.target, args.LDS_type, \
																					  args.da_strat, per_class_acc_before, acc_before)
	print(out_str)
	
	################################################################################################################
	#### Unsupervised adaptation of source model to target
	################################################################################################################
	da_file = '{:s}_{:s}_{}_{}_net_{:s}_{:s}_{:s}.pth'.format(args.id, args.da_strat, args.da_lr, args.cnn, \
															  args.source, args.target, args.LDS_type)
	outdir = 'checkpoints'
	os.makedirs(os.path.join(outdir, args.da_strat), exist_ok=True)
	outfile = os.path.join(outdir, args.da_strat, da_file)

	model_name = 'AdaptNet'
	if args.load_da and os.path.exists(outfile):		
		print('Trained {} checkpoint found: {}, loading...\n'.format(args.da_strat, outfile))
		net = get_model(model_name, num_cls=num_classes, weights_init=outfile, model=args.cnn, \
						l2_normalize=args.l2_normalize, temperature=args.temperature)
		source_model_adapt = net.tgt_net
	else:
		net = get_model(model_name, model=args.cnn, num_cls=num_classes, src_weights_init=source_path, \
						l2_normalize=args.l2_normalize, temperature=args.temperature).to(device)
		print(net)
		print('Training {} {} model for {}->{}-LT ({})\n'.format(args.da_strat, args.cnn, args.source, args.target, args.LDS_type))
		
		opt_net = utils.generate_optimizer(net.tgt_net, args, mode='da')
		solver = get_solver(args.da_strat, net.tgt_net, src_train_loader, \
							target_train_loader, tgt_train_idx, opt_net, device, num_classes, args)

		for epoch in range(args.da_num_epochs):
			if args.pseudo_balance_target: 
				print('\nEpoch {}: Re-estimating probabilities for pseudo-balancing...'.format(epoch))
				# Approximately class-balance target dataloader using pseudolabels at the start of each epoch
				target_dset_copy = copy.deepcopy(target_dset)
				src_train_dset_copy = copy.deepcopy(src_train_loader.dataset)

				_, gtlabels, plabels = utils.get_embedding(solver.net, target_train_loader, device, num_classes, args)

				target_dset_copy.train_dataset.targets_copy = copy.deepcopy(target_dset_copy.train_dataset.targets) # Create backup of actual labels				
				target_dset_copy.train_dataset.targets = plabels

				tgt_train_loader_pbalanced, _, _, _ = target_dset_copy.get_loaders(class_balance_train=True)
				tgt_train_loader_pbalanced.dataset.targets_copy = target_dset_copy.train_dataset.targets_copy
				solver.tgt_loader = tgt_train_loader_pbalanced					

			if args.da_strat == 'dann':
				opt_dis = utils.generate_optimizer(net.discriminator, args, mode='da')
				solver.solve(epoch, net.discriminator, opt_dis)
			else:
				solver.solve(epoch)

		print('Saving to', outfile)
		net.save(outfile)
		source_model_adapt = net.tgt_net

	# Evaluate adapted model
	print('\nEvaluating adapted model on {} test set'.format(args.target))	
	acc_after, cm_after = utils.test(source_model_adapt, device, target_test_loader, split="test", num_classes=num_classes)
	per_class_acc_after = cm_after.diagonal().numpy() / cm_after.sum(axis=1).numpy()
	per_class_acc_after = per_class_acc_after.mean() * 100
	
	print('###################################')
	out_str = '{}->{}-LT ({}), Before {}:\t Avg. acc={:.2f}%\tAgg. acc={:.2f}%'.format(args.source, args.target, args.LDS_type, \
																						 args.da_strat, per_class_acc_before, acc_before)
	out_str += '\n\t\t\tAfter {}:\t Avg. acc={:.2f}%\tAgg. acc={:.2f}%'.format(args.da_strat, per_class_acc_after, acc_after)
	print(out_str)	
	
	utils.plot_accuracy_statistics(cm_before, cm_after, num_classes, args, target_train_loader)

if __name__ == '__main__':
	main()