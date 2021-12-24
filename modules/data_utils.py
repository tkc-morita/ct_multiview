
# coding: utf-8

import numpy as np
import pandas as pd
import torch
import torchvision.datasets, torchvision.transforms
import itertools, copy, bisect, os.path, sys


def make_data_sampler(dataset, shuffle, seed=111):
	"""
	Simplified from maskrcnn_benchmark.data.build
	https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/build.py
	"""
	if shuffle:
		sampler = RandomSampler(dataset, seed=seed)
	else:
		sampler = torch.utils.data.sampler.SequentialSampler(dataset)
	return sampler

def make_batch_data_sampler(sampler, batch_size, num_iters=None, start_iter=0):
	"""
	Simplified from maskrcnn_benchmark.data.build
	https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/data/build.py
	"""
	batch_sampler = torch.utils.data.sampler.BatchSampler(
			sampler, batch_size, drop_last=False
		)
	if num_iters is not None:
		batch_sampler = IterationBasedBatchSampler(
			batch_sampler, num_iters, start_iter
		)
	return batch_sampler

class Renderer(object):
	def __init__(self, engine='mpl'):
		if engine=='mpl':
			from . import mpl_renderer as renderer
		elif engine=='pyvista':
			from . import pyvista_renderer as renderer
		elif engine=='pyrender':
			from . import pyrender_renderer as renderer
		elif engine=='plotly':
			from . import plotly_renderer as renderer
		self._load_stl = renderer.load_stl
		self._project2D = renderer.project2D

	def project2D(self, mesh, elev, azim, linewidths=0.1, alpha=1.0):
		img = self._project2D(mesh, elev, azim, linewidths=linewidths, alpha=alpha)
		img = img.convert("L") # To gray scale
		bbox = img.getbbox() # non-zero (non-black) regions.
		img = img.crop(bbox)
		return img

	def load_stl(self, path):
		return self._load_stl(path)

class ThreeDSurfaceDataset(object):
	def __init__(
			self,
			ann_file,
			root,
			target_col,
			path_col,
			elevs=[30,-30],
			num_azims=8,
			transforms=None,
			target2ix=None,
			img_line_width=0.1,
			img_alpha=1.0,
			noise_range=10.0,
			rendering_engine='mpl'
		):
		self.df_ann = pd.read_csv(ann_file)
		self.root = root
		self.target_col = target_col
		self.df_ann[self.target_col] = self.df_ann[self.target_col].astype(str)
		self.path_col = path_col
		self.transforms = transforms
		if target2ix is None:
			target2ix = self._create_target_to_ix()
		self.set_target_to_ix(target2ix)
		self.img_line_width = img_line_width
		self.img_alpha = img_alpha
		self.elevs = elevs
		self.azims_center = np.arange(num_azims) * 360.0 / num_azims
		self.noise_range = noise_range
		self.renderer = Renderer(rendering_engine)

	def _create_target_to_ix(self):
		target2ix = {target:ix for ix,target in enumerate(sorted(self.df_ann[self.target_col].unique()))}
		return target2ix

	def set_target_to_ix(self, target2ix):
		self.target2ix = target2ix
		self.ix2target = {ix:target for target,ix in self.target2ix.items()}

	def __getitem__(self, data_ix):
		sub_df = self.df_ann.iloc[[data_ix],:]

		mesh = self.renderer.load_stl(os.path.join(self.root, sub_df[self.path_col].iat[0]))
		azims,shift = self.sample_azims()
		imgs = [self.renderer.project2D(mesh, elev, azim, linewidths=self.img_line_width, alpha=self.img_alpha)
				for elev in self.elevs for azim in azims]

		target_ix = self.target2ix[sub_df[self.target_col].iat[0]]

		if self.transforms:
			imgs = [self.transforms(img) for img in imgs]

		return imgs, target_ix, shift, data_ix

	# def seed_random_state(self, seed=111):
		# seed = seed % 2**32
		# self.rng.manual_seed(seed)
		# pass

	def sample_azims(self):
		shift = torch.rand(1).item() * self.noise_range - 0.5 * self.noise_range
		azims = self.azims_center + shift
		return azims,shift

	def get_num_categories(self):
		return len(self.target2ix)

	def __len__(self):
		return self.df_ann.shape[0]

class RandomSampler(torch.utils.data.RandomSampler):
	"""
	Custom random sampler for iteration-based learning.
	"""
	def __init__(self, *args, seed=111, **kwargs):
		super(RandomSampler, self).__init__(*args, **kwargs)
		self.epoch = 0
		self.start_ix = 0
		self.seed = seed

	def set_epoch(self, epoch):
		self.epoch = epoch

	def set_start_ix(self, start_ix):
		self.start_ix = start_ix

	def __iter__(self):
		g = torch.Generator()
		g.manual_seed(self.epoch+self.seed)
		start_ix = self.start_ix
		self.start_ix = 0
		n = len(self.data_source)
		if self.replacement:
			return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64, generator=g).tolist()[start_ix:])
		return iter(torch.randperm(n, generator=g).tolist()[start_ix:])

class IterationBasedBatchSampler(torch.utils.data.sampler.BatchSampler):
	"""
	Modified from maskrcnn_benchmark.data.samplers.iteration_based_batch_sampler:
	https://github.com/facebookresearch/maskrcnn-benchmark

	Wraps a BatchSampler, resampling from it until
	a specified number of iterations have been sampled
	"""

	def __init__(self, batch_sampler, num_iterations, start_iter=0):
		self.batch_sampler = batch_sampler
		self.num_iterations = num_iterations
		self.start_iter = start_iter
		start_ix = (self.start_iter % len(self.batch_sampler)) * self.batch_sampler.batch_size
		self.batch_sampler.sampler.set_start_ix(start_ix)
		# global iteration
		# iteration = 0

	def __iter__(self):
		global iteration
		iteration = self.start_iter
		epoch = iteration // len(self.batch_sampler)
		while iteration <= self.num_iterations:
			if hasattr(self.batch_sampler.sampler, "set_epoch"):
				self.batch_sampler.sampler.set_epoch(epoch)
			for batch in self.batch_sampler:
				iteration += 1
				if iteration > self.num_iterations:
					break
				yield batch
			epoch += 1

	def __len__(self):
		return self.num_iterations


def collator(batch):
	imgs, target_ixs, shift, data_ixs = zip(*batch)
	imgs = [torch.stack(view, dim=0) for view in zip(*imgs)]
	target_ixs = torch.tensor(target_ixs)
	return imgs, target_ixs, shift, data_ixs

# def worker_init_fn(worker_ix):
	# worker_info = torch.utils.data.get_worker_info()
	# dataset = worker_info.dataset  # the dataset copy in this worker process
	# dataset.seed_random_state(worker_info.seed)

def make_data_loader(dataset, batch_size, is_train=True, start_iter=0, num_workers=1, num_iters=None, random_seed=111):
	if is_train:
		shuffle = True
		assert not num_iters is None, 'num_iters must be specified in training.'
	else:
		shuffle = False
		num_iters = None
		start_iter = 0

	sampler = make_data_sampler(dataset, shuffle, random_seed)
	batch_sampler = make_batch_data_sampler(
		sampler, batch_size, num_iters, start_iter
	)
	data_loader = torch.utils.data.DataLoader(
		dataset,
		num_workers=num_workers,
		batch_sampler=batch_sampler,
		collate_fn=collator,
		# worker_init_fn=worker_init_fn,
	)
	return data_loader


class ProportionalResizeAndPad(object):
	"""
	Resize PIL images to a square, keeping the aspect ratio with padding the shorter edge.
	"""
	def __init__(self, size):
		self.size = size

	def __call__(self, img):
		# Step 1: rescale img s.t. the length of the LONGER edge matches self.size
		w,h = img.size
		if w < h:
			oh = self.size
			ow = int(oh * w / h)
			pad_len = (oh-ow)*0.5
			left_pad = int(np.floor(pad_len))
			right_pad = int(np.ceil(pad_len))
			padding = (left_pad, 0, right_pad, 0)
		else:
			ow = self.size
			oh = int(ow * h / w)
			pad_len = (ow-oh)*0.5
			top_pad = int(np.floor(pad_len))
			bottom_pad = int(np.ceil(pad_len))
			padding = (0, top_pad, 0, bottom_pad)
		img = img.resize((ow, oh))
		# Step 2: Pad the shorter edge with 0s.
		img = torchvision.transforms.functional.pad(img, padding, fill=0, padding_mode='constant')
		return img
