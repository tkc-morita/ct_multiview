# coding: utf-8

import torch
import torchvision
import math

class MultiViewClassifier(torch.nn.Module):
	def __init__(
			self,
			num_categories,
			num_views,
			cnn='resnet50',
			aggregator='cnn',
			in_channels=4,
			freeze_at=None,
			transformer_kwargs=dict(),
			cnn_kwargs=dict()):
		super(MultiViewClassifier, self).__init__()
		self.cnn_kwargs = cnn_kwargs
		self.transformer_kwargs = transformer_kwargs
		self.cnn_name = cnn
		self.num_views = num_views
		self.view_encoder = CNNwoFC(cnn, in_channels, freeze_at, **cnn_kwargs)
		self.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
		descriptor_out = self.view_encoder.net.fc.in_features
		self.aggregator_type = aggregator
		if aggregator=='cnn':
			self.aggregator = ViewPool()
			self.fc = self.view_encoder.net.fc
		elif aggregator=='transformer':
			self.aggregator = TransformerAggregator(descriptor_out, **transformer_kwargs)
			self.fc = torch.nn.Linear(self.aggregator.get_outsize(), num_categories)
		else:
			raise ValueError('aggregator must be either "cnn" or "transformer"')

	def forward(self, views):
		views = self.get_view_features(views)
		views = [self.avgpool(v) for v in views]
		views = torch.stack(views, dim=0)
		features = self.aggregator(views)
		logits = self.fc(features)
		return logits
	
	def get_view_features(self, views):
		views = [self.view_encoder(v) for v in views]
		return views

	def pack_init_args(self):
		args = {
			"num_categories":self.fc.out_features,
			"num_views":self.num_views,
			"cnn":self.cnn_name,
			"freeze_at":self.view_encoder.freeze_at,
			"in_channels":self.view_encoder.net.conv1.in_channels,
			"aggregator":self.aggregator_type,
			"transformer_kwargs":self.transformer_kwargs,
			"cnn_kwargs":self.cnn_kwargs
		}
		return args

class CNNwoFC(torch.nn.Module):
	def __init__(self, architecture, in_channels, freeze_at=None, **kwargs):
		super(CNNwoFC, self).__init__()
		assert architecture.startswith('resnet'), 'Only Resnet is currently supported.'
		self.architecture = architecture
		self.net = getattr(torchvision.models, architecture)(pretrained=True, **kwargs)
		self._reset_input_channels(in_channels)
		self.freeze_layers(freeze_at)

	def forward(self, x):
		x = self.net.conv1(x)
		x = self.net.bn1(x)
		x = self.net.relu(x)
		x = self.net.maxpool(x)

		x = self.net.layer1(x)
		x = self.net.layer2(x)
		x = self.net.layer3(x)
		x = self.net.layer4(x)
		return x

	def _reset_input_channels(self, in_channels):
		self.net.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

	def _freeze_parameters(self, module):
		for param in module.parameters():
			param.requires_grad = False

	def freeze_layers(self, freeze_at):
		self.freeze_at = freeze_at
		if not self.freeze_at is None:
			self._freeze_parameters(self.net.conv1)
			self._freeze_parameters(self.net.bn1)
			for layer in range(1, self.freeze_at):
				self._freeze_parameters(getattr(self.net, 'layer{layer}'.format(layer=layer)))

class ViewPool(torch.nn.Module):
	def forward(self, views):
		return views.max(0)[0].view(views.size(1),-1)

class TransformerAggregator(torch.nn.Module):
	def __init__(self, view_channels, d_model, nheads, num_layers, dim_feedforward=256):
		super(TransformerAggregator, self).__init__()
		self.linear = torch.nn.Linear(view_channels, d_model)
		self.d_model = d_model
		layer = torch.nn.TransformerEncoderLayer(d_model,nheads,dim_feedforward)
		self.transformer = torch.nn.TransformerEncoder(layer, num_layers)

	def forward(self, views):
		V,B,C,_,_ = views.size()
		views = views.view(-1,C)
		views = self.linear(views)
		views = views.view(V,B,-1)
		out = self.transformer(views)
		return out[0]

	def get_outsize(self):
		return self.d_model