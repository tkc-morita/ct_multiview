# coding: utf-8

import torch
import torchvision
import numpy as np
import pandas as pd
from modules import data_utils
import learning
import os, argparse, itertools, json


class Predictor(learning.Learner):
	def __init__(self, model_config_path, device = 'cpu'):
		self.device = torch.device(device)
		self.retrieve_model(checkpoint_path = model_config_path, device=device)
		self.model.to(self.device)
		for param in self.model.parameters():
			param.requires_grad = False
		self.model.eval()


	def predict(self, views, class_ix):
		views = [v.to(self.device) for v in views]
		class_ix = class_ix.to(self.device)
		logits = self.model(views)
		max_probs,map_classes = torch.nn.functional.softmax(logits, -1).max(-1)
		class_matches = map_classes==class_ix
		return max_probs.data.cpu().numpy(),map_classes.data.cpu().numpy(),class_matches.data.cpu().numpy()


	def predict_dataset(
		self,
		dataloader,
		save_path,
		samples_per_data=1
		):
		save_dir = os.path.dirname(save_path)
		if not os.path.isdir(save_dir):
			os.makedirs(save_dir)
		for epoch in range(samples_per_data):
			for views, class_ixs, azim_shift, data_ixs in dataloader:
				max_probs,map_classes,class_matches = self.predict(views, class_ixs)
				df_ann = dataloader.dataset.df_ann.loc[data_ixs,:]
				df_ann['azim_shift'] = azim_shift
				df_ann['max_prob'] = max_probs
				df_ann['map_class'] = map_classes
				df_ann['correct_prediction'] = class_matches.astype(bool)
				df_ann['sample_ix'] = epoch

				if os.path.isfile(save_path):
					df_ann.to_csv(save_path, index=False, mode='a', header=False)
				else:
					df_ann.to_csv(save_path, index=False)



def get_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('model_weight', type=str, help='Path to the checkpoint of the pretrained model.')
	parser.add_argument('input_root', type=str, help='Path to the root annotationy under which inputs are located.')
	parser.add_argument('annotation_file', type=str, help='Path to the annotation csv.')
	parser.add_argument('save_path', type=str, help='Path to the csv where results are saved.')
	parser.add_argument('-d', '--device', type=str, default='cpu', help='Computing device.')
	parser.add_argument('--class_col', type=str, default='sex', help='Column in the annotation csv for the target label.')
	parser.add_argument('--path_col', type=str, default='stl_path', help='Column in the annotation csv for the relative path to the STL data.')
	parser.add_argument('--input_edge_size', type=int, default=224, help='The height and width of the resized bboxed image.')
	parser.add_argument('--img_line_width', type=float, default=0.01, help='Line width of surface images.')
	parser.add_argument('--img_alpha', type=float, default=0.1, help='Opacity of surface images.')
	parser.add_argument('--elevs', type=float, nargs='+', default=[30.0, -30.0], help='Elevation of the views [-90,90].')
	parser.add_argument('--num_azims', type=int, default=8, help='# of view azimuths.')
	parser.add_argument('--noise_range', type=float, default=0.0, help='Range of noise on the azimuths [0,180).')
	parser.add_argument('--samples_per_data', type=int, default=1, help='# of samples per data.')
	parser.add_argument('-s', '--seed', type=int, default=1111, help='random seed')
	parser.add_argument('-b', '--batch_size', type=int, default=512, help='Batch size for training.')
	parser.add_argument('--num_workers', type=int, default=1, help='# of workers (>=1).')
	parser.add_argument('--rendering_engine', type=str, default='mpl', help='3D rendering engine to use. "mpl" (for matplotlib) or "pyvista".')

	return parser.parse_args()


if __name__ == '__main__':
	args = get_args()

	# Get a model.
	predictor = Predictor(args.model_weight, device = args.device)

	class2ix_path = os.path.join(os.path.dirname(args.model_weight), 'target2ix.json')
	with open(class2ix_path, 'r') as f:
		class2ix = json.load(f)

	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	transforms = [
		data_utils.ProportionalResizeAndPad(args.input_edge_size),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5,), (0.5,))
	]

	dataset = data_utils.ThreeDSurfaceDataset(
		args.annotation_file,
		args.input_root,
		args.class_col,
		args.path_col,
		elevs=args.elevs,
		num_azims=args.num_azims,
		transforms=torchvision.transforms.Compose(transforms),
		target2ix=class2ix,
		img_line_width=args.img_line_width,
		img_alpha=args.img_alpha,
		noise_range=args.noise_range,
		rendering_engine=args.rendering_engine
	)

	dataloader = data_utils.make_data_loader(dataset, args.batch_size, is_train=False, num_workers=args.num_workers)

	# Predict
	predictor.predict_dataset(
		dataloader,
		args.save_path,
		samples_per_data=args.samples_per_data
		)