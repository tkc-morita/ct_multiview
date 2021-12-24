# coding: utf-8

import torch

class ScoreCAM(object):
	"""
	Score-CAM proposed by Wang et al. (2019).
	https://arxiv.org/abs/1910.01279
	"""
	def __init__(self, model, normalize_per_view=False):
		self.model = torch.nn.DataParallel(ParallelModel(model))
		self.act2prob = ActMap2Probs(model)
		self.normalize_per_view = normalize_per_view

	def to(self, device):
		self.model.to(device)
		self.act2prob.to(device)

	def eval(self):
		self.model.eval()
		self.act2prob.eval()

	def __call__(self, views):
		"""
		Compare two viewss of DIFFERENT classes.
		"""
		with torch.no_grad():
			# views = views[None,:,:,:] # Add the batch dimension.
			views = torch.stack(views, dim=0) # num_views x num_channels x H x W
			act_maps = self.model(views, only_act_maps=True)
			probs = self.act2prob(act_maps)
			act_maps = torch.nn.functional.interpolate(
						act_maps, views.size()[-2:],
						mode="bilinear", align_corners=False
						)
			mask = self.normalize(act_maps)
			
			masked = views[:,None,:,:,:] * mask[:,:,None,:,:]
			num_views,num_act_maps,C,H,W = masked.size()
			logits = [self.model(m) for m in masked]
			if self.normalize_per_view:
				logits = torch.stack(logits, dim=0)
				weights = torch.nn.functional.softmax(logits, dim=1)
			else:
				logits = torch.cat(logits, dim=0)
				weights = torch.nn.functional.softmax(logits, dim=0)
				weights = weights.view(num_views,num_act_maps,-1)

			score_cam = torch.einsum('vahw,vac->vhwc', act_maps, weights).relu()
			if self.normalize_per_view:
				normalizer = score_cam.view(num_views,-1,score_cam.size(-1)).max(1)[0][:,None,None,:]
			else:
				normalizer = score_cam.view(-1,score_cam.size(-1)).max(0)[0][None,None,None,:]
			normalizer = normalizer.masked_fill(normalizer<=0,1.0)
			score_cam = score_cam / normalizer
		return score_cam, probs



	@staticmethod
	def normalize(act_maps):
		V,C,_,_ = act_maps.size()
		flatten = act_maps.view(V*C,-1)
		min_ = flatten.min(-1)[0]
		denominator = flatten.max(-1)[0] - min_
		denominator = denominator.masked_fill(denominator<=0, 1.0)
		act_maps = (act_maps-min_.view(V,C,1,1)) / denominator.view(V,C,1,1)
		return act_maps

class ParallelModel(torch.nn.Module):
	def __init__(self, model):
		super(ParallelModel, self).__init__()
		self.model = model

	def forward(self, x, only_act_maps=False):
		features = self.model.view_encoder(x)
		if only_act_maps:
			return features
		features = self.model.avgpool(features)
		logits = self.model.fc(features.view(features.size(0),-1))
		return logits

class ActMap2Probs(torch.nn.Module):
	def __init__(self, model):
		super(ActMap2Probs, self).__init__()
		self.avgpool = model.avgpool
		self.aggregator = model.aggregator
		self.fc = model.fc

	def forward(self, act_maps):
		features = self.avgpool(act_maps)
		features = self.aggregator(features[:,None,:]).squeeze(0)
		logits = self.fc(features)
		probs = logits.softmax(-1)
		return probs