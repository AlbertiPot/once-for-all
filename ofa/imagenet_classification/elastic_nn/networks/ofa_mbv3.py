# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import random

from ofa.imagenet_classification.elastic_nn.modules.dynamic_layers import DynamicMBConvLayer
from ofa.utils.layers import ConvLayer, IdentityLayer, LinearLayer, MBConvLayer, ResidualBlock
from ofa.imagenet_classification.networks import MobileNetV3
from ofa.utils import make_divisible, val2list, MyNetwork

__all__ = ['OFAMobileNetV3']


class OFAMobileNetV3(MobileNetV3):

	def __init__(self, n_classes=1000, bn_param=(0.1, 1e-5), dropout_rate=0.1, base_stage_width=None, width_mult=1.0,
	             ks_list=3, expand_ratio_list=6, depth_list=4):

		self.width_mult = width_mult
		self.ks_list = val2list(ks_list, 1)
		self.expand_ratio_list = val2list(expand_ratio_list, 1)
		self.depth_list = val2list(depth_list, 1)

		self.ks_list.sort()
		self.expand_ratio_list.sort()
		self.depth_list.sort()

		base_stage_width = [16, 16, 24, 40, 80, 112, 160, 960, 1280]

		final_expand_width = make_divisible(base_stage_width[-2] * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)
		last_channel = make_divisible(base_stage_width[-1] * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)

		stride_stages = [1, 2, 2, 2, 1, 2]
		act_stages = ['relu', 'relu', 'relu', 'h_swish', 'h_swish', 'h_swish']
		se_stages = [False, False, True, False, True, True]
		n_block_list = [1] + [max(self.depth_list)] * 5															# 共有6个stage，第一个stage仅有1个block [1,4,4,4,4,4]
		width_list = []
		for base_width in base_stage_width[:-2]:
			width = make_divisible(base_width * self.width_mult, MyNetwork.CHANNEL_DIVISIBLE)					# 除去倒数的两个with，其余乘上width乘子后仍然要被8整除
			width_list.append(width)

		input_channel, first_block_dim = width_list[0], width_list[1]
		# first conv layer
		first_conv = ConvLayer(3, input_channel, kernel_size=3, stride=2, act_func='h_swish')
		
		# first block
		first_block_conv = MBConvLayer(																			# 第一个block中的MB卷积层
			in_channels=input_channel, out_channels=first_block_dim, kernel_size=3, stride=stride_stages[0],
			expand_ratio=1, act_func=act_stages[0], use_se=se_stages[0],
		)
		first_block = ResidualBlock(																			# 第一个block：并联一个卷积层+恒等映射层构成残差模块
			first_block_conv,
			IdentityLayer(first_block_dim, first_block_dim) if input_channel == first_block_dim else None,
		)

		# inverted residual blocks
		self.block_group_info = []
		blocks = [first_block]
		_block_index = 1
		feature_dim = first_block_dim

		for width, n_block, s, act_func, use_se in zip(width_list[2:], n_block_list[1:],
		                                               stride_stages[1:], act_stages[1:], se_stages[1:]):
			self.block_group_info.append([_block_index + i for i in range(n_block)])							# block_group_info存储了全部inverted blocks中的层数: [[1,2,3,4], [5,6,7,8], ... ] 
			_block_index += n_block

			output_channel = width
			for i in range(n_block):
				if i == 0:																						# 每个block中的第一个层的stride为strid_list中指定的s，其余全部为1
					stride = s
				else:
					stride = 1
				mobile_inverted_conv = DynamicMBConvLayer(
					in_channel_list=val2list(feature_dim), out_channel_list=val2list(output_channel),
					kernel_size_list=ks_list, expand_ratio_list=expand_ratio_list,								# 虽然传入的是ks_list和expand_ratio_list, 但是在最低层dynamic_op.py中是用最大的kernel size和最大expand_ratio初始化
					stride=stride, act_func=act_func, use_se=use_se,
				)
				if stride == 1 and feature_dim == output_channel:
					shortcut = IdentityLayer(feature_dim, feature_dim)
				else:
					shortcut = None
				blocks.append(ResidualBlock(mobile_inverted_conv, shortcut))
				feature_dim = output_channel
		
		# final expand layer
		final_expand_layer = ConvLayer(feature_dim, final_expand_width, kernel_size=1, act_func='h_swish')
		
		# feature mix layer
		feature_mix_layer = ConvLayer(
			final_expand_width, last_channel, kernel_size=1, bias=False, use_bn=False, act_func='h_swish',
		)

		# classifier
		classifier = LinearLayer(last_channel, n_classes, dropout_rate=dropout_rate)

		super(OFAMobileNetV3, self).__init__(first_conv, blocks, final_expand_layer, feature_mix_layer, classifier)	# 调用mobilenetv3类中的初始化方法初始化一个实例

		# set bn param
		self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

		# runtime_depth
		self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]								# 重要!!! runtime_depth是运行时各个stage的激活的blocks的数量

	""" MyNetwork required methods """

	@staticmethod
	def name():
		return 'OFAMobileNetV3'

	def forward(self, x):
		# first conv
		x = self.first_conv(x)
		# first block
		x = self.blocks[0](x)
		# blocks
		for stage_id, block_idx in enumerate(self.block_group_info):												# 上文block_group_info：[[1,2,3,4], [5,6,7,8], ... ], 共有5个choice stage以stage_id索引，每个stage有4个choice block，以block_idx索引
			
			depth = self.runtime_depth[stage_id]																	# 根据每个stage的激活的blocks数目n，选择要过的top-n blocks
			active_idx = block_idx[:depth]
			
			for idx in active_idx:																					# 根据激活的block_idx逐个过数据
				x = self.blocks[idx](x)
		x = self.final_expand_layer(x)
		x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
		x = self.feature_mix_layer(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

	@property
	def module_str(self):
		_str = self.first_conv.module_str + '\n'
		_str += self.blocks[0].module_str + '\n'

		for stage_id, block_idx in enumerate(self.block_group_info):
			depth = self.runtime_depth[stage_id]
			active_idx = block_idx[:depth]
			for idx in active_idx:
				_str += self.blocks[idx].module_str + '\n'

		_str += self.final_expand_layer.module_str + '\n'
		_str += self.feature_mix_layer.module_str + '\n'
		_str += self.classifier.module_str + '\n'
		return _str

	@property
	def config(self):
		return {
			'name': OFAMobileNetV3.__name__,
			'bn': self.get_bn_param(),
			'first_conv': self.first_conv.config,
			'blocks': [
				block.config for block in self.blocks
			],
			'final_expand_layer': self.final_expand_layer.config,
			'feature_mix_layer': self.feature_mix_layer.config,
			'classifier': self.classifier.config,
		}

	@staticmethod
	def build_from_config(config):
		raise ValueError('do not support this function')

	@property
	def grouped_block_index(self):
		return self.block_group_info

	def load_state_dict(self, state_dict, **kwargs):
		model_dict = self.state_dict()
		for key in state_dict:
			if '.mobile_inverted_conv.' in key:
				new_key = key.replace('.mobile_inverted_conv.', '.conv.')
			else:
				new_key = key
			if new_key in model_dict:
				pass
			elif '.bn.bn.' in new_key:
				new_key = new_key.replace('.bn.bn.', '.bn.')
			elif '.conv.conv.weight' in new_key:
				new_key = new_key.replace('.conv.conv.weight', '.conv.weight')
			elif '.linear.linear.' in new_key:
				new_key = new_key.replace('.linear.linear.', '.linear.')
			##############################################################################
			elif '.linear.' in new_key:
				new_key = new_key.replace('.linear.', '.linear.linear.')
			elif 'bn.' in new_key:
				new_key = new_key.replace('bn.', 'bn.bn.')
			elif 'conv.weight' in new_key:
				new_key = new_key.replace('conv.weight', 'conv.conv.weight')
			else:
				raise ValueError(new_key)
			assert new_key in model_dict, '%s' % new_key
			model_dict[new_key] = state_dict[key]
		super(OFAMobileNetV3, self).load_state_dict(model_dict)

	""" set, sample and get active sub-networks """

	def set_max_net(self):
		self.set_active_subnet(ks=max(self.ks_list), e=max(self.expand_ratio_list), d=max(self.depth_list))

	def set_active_subnet(self, ks=None, e=None, d=None, **kwargs):
		ks = val2list(ks, len(self.blocks) - 1)
		expand_ratio = val2list(e, len(self.blocks) - 1)
		depth = val2list(d, len(self.block_group_info))

		for block, k, e in zip(self.blocks[1:], ks, expand_ratio):
			if k is not None:
				block.conv.active_kernel_size = k																		# 设置运行时的ks
			if e is not None:
				block.conv.active_expand_ratio = e																		# 设置运行时的expand ratio

		for i, d in enumerate(depth):
			if d is not None:
				self.runtime_depth[i] = min(len(self.block_group_info[i]), d)											# 设置每个stage运行时的block数								

	def set_constraint(self, include_list, constraint_type='depth'):
		if constraint_type == 'depth':
			self.__dict__['_depth_include_list'] = include_list.copy()
		elif constraint_type == 'expand_ratio':
			self.__dict__['_expand_include_list'] = include_list.copy()
		elif constraint_type == 'kernel_size':
			self.__dict__['_ks_include_list'] = include_list.copy()
		else:
			raise NotImplementedError

	def clear_constraint(self):
		self.__dict__['_depth_include_list'] = None
		self.__dict__['_expand_include_list'] = None
		self.__dict__['_ks_include_list'] = None

	def sample_active_subnet(self):
		ks_candidates = self.ks_list if self.__dict__.get('_ks_include_list', None) is None \
			else self.__dict__['_ks_include_list']																		# 当constrain是none时，在原始的ks_list中采样，若限制list非空，在受限条件下采样
		expand_candidates = self.expand_ratio_list if self.__dict__.get('_expand_include_list', None) is None \
			else self.__dict__['_expand_include_list']
		depth_candidates = self.depth_list if self.__dict__.get('_depth_include_list', None) is None else \
			self.__dict__['_depth_include_list']

		# sample kernel size
		ks_setting = []
		if not isinstance(ks_candidates[0], list):
			ks_candidates = [ks_candidates for _ in range(len(self.blocks) - 1)]										# self.blocks共5个6个stage，第一个为1个block，其余都是4个stage。此处是将kslist [3,5,7]为21个block每个复制一次：[[3,5,7]×21]
		for k_set in ks_candidates:																						# 遍历每个block对应的ks_list
			k = random.choice(k_set)																					# 从每个block对应的list[3,5,7]采样一个ks
			ks_setting.append(k)

		# sample expand ratio
		expand_setting = []
		if not isinstance(expand_candidates[0], list):
			expand_candidates = [expand_candidates for _ in range(len(self.blocks) - 1)]
		for e_set in expand_candidates:
			e = random.choice(e_set)
			expand_setting.append(e)

		# sample depth
		depth_setting = []
		if not isinstance(depth_candidates[0], list):
			depth_candidates = [depth_candidates for _ in range(len(self.block_group_info))]
		for d_set in depth_candidates:
			d = random.choice(d_set)
			depth_setting.append(d)

		self.set_active_subnet(ks_setting, expand_setting, depth_setting)												# 将随机采样的ks, e,和depth设置
		return {
			'ks': ks_setting,
			'e': expand_setting,
			'd': depth_setting,
		}

	def get_active_subnet(self, preserve_weight=True):
		first_conv = copy.deepcopy(self.first_conv)
		blocks = [copy.deepcopy(self.blocks[0])]

		final_expand_layer = copy.deepcopy(self.final_expand_layer)
		feature_mix_layer = copy.deepcopy(self.feature_mix_layer)
		classifier = copy.deepcopy(self.classifier)

		input_channel = blocks[0].conv.out_channels
		# blocks
		for stage_id, block_idx in enumerate(self.block_group_info):													# [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
			depth = self.runtime_depth[stage_id]
			active_idx = block_idx[:depth]
			stage_blocks = []
			for idx in active_idx:
				stage_blocks.append(ResidualBlock(
					self.blocks[idx].conv.get_active_subnet(input_channel, preserve_weight),							# self.blocks[idx]是ResidualBlock， .conv指向DynamicMBConvLayer，其中包含一个类函数get_active_subnet()
					copy.deepcopy(self.blocks[idx].shortcut)
				))
				input_channel = stage_blocks[-1].conv.out_channels
			blocks += stage_blocks

		_subnet = MobileNetV3(first_conv, blocks, final_expand_layer, feature_mix_layer, classifier)					# 除了first block都是deep copy来的， 其余的5个stage的choice blocks都是调用激活的运行时子网
		_subnet.set_bn_param(**self.get_bn_param())
		return _subnet

	def get_active_net_config(self):
		# first conv
		first_conv_config = self.first_conv.config
		first_block_config = self.blocks[0].config
		final_expand_config = self.final_expand_layer.config
		feature_mix_layer_config = self.feature_mix_layer.config
		classifier_config = self.classifier.config

		block_config_list = [first_block_config]
		input_channel = first_block_config['conv']['out_channels']
		for stage_id, block_idx in enumerate(self.block_group_info):
			depth = self.runtime_depth[stage_id]
			active_idx = block_idx[:depth]
			stage_blocks = []
			for idx in active_idx:
				stage_blocks.append({
					'name': ResidualBlock.__name__,
					'conv': self.blocks[idx].conv.get_active_subnet_config(input_channel),
					'shortcut': self.blocks[idx].shortcut.config if self.blocks[idx].shortcut is not None else None,
				})
				input_channel = self.blocks[idx].conv.active_out_channel
			block_config_list += stage_blocks

		return {
			'name': MobileNetV3.__name__,
			'bn': self.get_bn_param(),
			'first_conv': first_conv_config,
			'blocks': block_config_list,
			'final_expand_layer': final_expand_config,
			'feature_mix_layer': feature_mix_layer_config,
			'classifier': classifier_config,
		}

	""" Width Related Methods """

	def re_organize_middle_weights(self, expand_ratio_stage=0):
		for block in self.blocks[1:]:
			block.conv.re_organize_middle_weights(expand_ratio_stage)
