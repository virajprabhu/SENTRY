
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from .models import register_model, get_model

@register_model('AdaptNet')
class AdaptNet(nn.Module):
	"Defines an Adapt Network."
	def __init__(self, num_cls=10, model='LeNet', src_weights_init=None, weights_init=None, \
				 weight_sharing=None, l2_normalize=False, temperature=1.0):
		super(AdaptNet, self).__init__()
		self.name = 'AdaptNet'
		self.base_model = model	
		
		self.num_cls = num_cls
		self.cls_criterion = nn.CrossEntropyLoss()
		self.gan_criterion = nn.CrossEntropyLoss()
		self.weight_sharing = weight_sharing
		
		self.l2_normalize = l2_normalize
		self.temperature = temperature
		
		self.setup_net()
		if weights_init is not None:
			self.load(weights_init)
		elif src_weights_init is not None:
			self.load_src_net(src_weights_init)
		else:
			raise Exception('AdaptNet must be initialized with weights.')
	
	def custom_copy(self, src_net, weight_sharing):
		tgt_net = copy.deepcopy(src_net)
		if weight_sharing != 'None':
			if weight_sharing == 'classifier': tgt_net.classifier = src_net.classifier
			elif weight_sharing == 'full': tgt_net = src_net
		return tgt_net
	
	def setup_net(self):
		"""Setup source, target and discriminator networks."""
		self.src_net = get_model(self.base_model, num_cls=self.num_cls, \
								 l2_normalize=self.l2_normalize, temperature=self.temperature)
		self.tgt_net = self.custom_copy(self.src_net, self.weight_sharing)

		input_dim = self.num_cls
		self.discriminator = nn.Sequential(
				nn.Linear(input_dim, 500),
				nn.ReLU(),
				nn.Linear(500, 500),
				nn.ReLU(),
				nn.Linear(500, 2),
				)

		self.image_size = self.src_net.image_size
		self.num_channels = self.src_net.num_channels

	def load(self, init_path):
		"Loads full src and tgt models."
		net_init_dict = torch.load(init_path, map_location=torch.device('cpu'))
		self.load_state_dict(net_init_dict)

	def load_src_net(self, init_path):
		"""Initialize source and target with source
		weights."""
		self.src_net.load(init_path)
		self.tgt_net.load(init_path)

	def save(self, out_path):
		torch.save(self.state_dict(), out_path)