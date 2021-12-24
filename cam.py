# coding: utf-8

import torch
import torchvision
import numpy as np
import pandas as pd
from modules import data_utils, score_cam
import learning
import os, argparse, itertools, json
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Predictor(learning.Learner):
	def __init__(self, model_config_path, device = 'cpu', use_grad_cam=False, normalize_per_view=False):
		self.device = torch.device(device)
		self.retrieve_model(checkpoint_path = model_config_path, device=device, only_load_model=True)
		if use_grad_cam:
			raise NotImplementedError
			# self.model.to(self.device)
			# self.gcam = grad_cam.GradCAM(model=self.model.cnn)
			# self.run_cam = self._run_grad_cam
		else:
			self.scam = score_cam.ScoreCAM(model=self.model, normalize_per_view=normalize_per_view)
			self.run_cam = self._run_score_cam
			for param in self.model.parameters():
				param.requires_grad = False
			self.scam.to(self.device)
			self.scam.eval()


	def predict(self, views):
		views = [v.to(self.device) for v in views]
		regions,probs = self.run_cam(views)
		return regions,probs

	def _run_score_cam(self, views):
		regions,probs = self.scam(views)
		return regions,probs

	# def _run_grad_cam(self, views, class_ix):
		# raise NotImplementedError
		# views = [v[None,...] for v in views]
		# _ = self.gcam.forward(views)
		# self.gcam.backward(class_ix.view(1,1))
		# regions = self.gcam.generate('layer4')[0,0] # 1st dimension is for intra-batch ixs, and 2nd for channels. Both only have one.
		# return regions,prob


	def predict_dataset(
		self,
		df_ann,
		path_col,
		class_col,
		data_root,
		save_root,
		input_edge_size,
		class2ix,
		elevs=[30],
		azims=np.arange(8) * 360.0 / 8,
		rendering_engine='mpl',
		img_line_width=0.1,
		img_alpha=1.0,
		):
		prop_resize_and_pad = data_utils.ProportionalResizeAndPad(input_edge_size)
		transforms = torchvision.transforms.Compose(
			[
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Normalize((0.5,), (0.5,))
			]
			)
		class_probs = []
		renderer = data_utils.Renderer(rendering_engine)
		# fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
		if not os.path.isdir(save_root):
			os.makedirs(save_root)
		for row_tpl in df_ann.itertuples(index=False):
			row_dict = row_tpl._asdict()
			mesh = renderer.load_stl(os.path.join(data_root, row_dict[path_col]))
			# triverts = np.load(os.path.join(data_root, row_dict[path_col]), mmap_mode='r')
			save_name = os.path.splitext(os.path.basename(row_dict[path_col]))[0]
			save_path = os.path.join(save_root, save_name)
			# video = cv2.VideoWriter(save_path, fourcc, 4.0, (input_edge_size, input_edge_size))

			# class_ = row_dict[class_col]
			# class_ix = class2ix[class_]
			# class_ix = torch.as_tensor(class_ix)
			raw_imgs = [prop_resize_and_pad(
						renderer.project2D(
							mesh, elev, azim,
							linewidths=img_line_width, alpha=img_alpha
						)
						)
						for elev in elevs for azim in azims]
			imgs = [transforms(img) for img in raw_imgs]

			regions,probs = self.predict(imgs)
			regions = regions.cpu().numpy()
			d = {'{cls}_{elev}_{azim}'.format(cls=cls_,elev=elev,azim=azim)
					:regions[view_ix,...,cls_ix]
				for cls_,cls_ix in class2ix.items()
				for view_ix,(elev,azim) in enumerate(itertools.product(elevs,azims))}
			np.savez(save_path+'.npz', **d)

			class_probs += [(row_dict[path_col],cls_,probs[cls_ix].item())
							for cls_,cls_ix in class2ix.items()]

			raw_imgs = np.array([np.array(img, dtype=np.uint8) for img in raw_imgs])[...,None] # PIL to cv2 and add the color dimension.
			overlaid = {cls_:self.overlay_cam(regions[...,cls_ix], raw_imgs)
						for cls_,cls_ix in class2ix.items()}
			nrows = len(elevs)
			ncols = len(azims)
			overlaid = {cls_:ol.reshape(nrows,ncols,*ol.shape[1:])
						for cls_,ol in overlaid.items()}
			for cls_,ol in overlaid.items():
				fig,axes = plt.subplots(
							nrows=nrows, ncols=ncols, squeeze=False,
							figsize=(ncols*2,nrows*2),
							)
				for row,elev,imgs_per_elev in zip(axes,elevs,ol):
					for ax,axim,img in zip(row,azims,imgs_per_elev):
						ax.imshow(img)
						ax.spines['right'].set_visible(False)
						ax.spines['top'].set_visible(False)
						ax.spines['bottom'].set_visible(False)
						ax.spines['left'].set_visible(False)
						ax.set_xticks([])
						ax.set_yticks([])
					row[0].set_ylabel('{:.1f}'.format(elev), fontsize='x-large')
				for ax,azim in zip(axes[0],azims):
					ax.set_title('{:.1f}'.format(azim), fontsize='x-large')
				# fig.suptitle('P({cls})={prob:0.3f} (ground truth: {gt})\nAzimuth'.format(
				# 	cls=cls_, gt=class2ix[row_dict[class_col]],
				# 	prob=probs[class2ix[cls_]]), va='baseline', fontsize='xx-large')
				fig.suptitle('Azimuth', fontsize='xx-large', va='baseline')
				fig.text(0.1, 0.5, 'Elevation', fontsize='xx-large', va='center', ha='center', rotation='vertical',)
				plt.savefig(save_path+'_predicting-{}.png'.format(cls_), bbox_inches='tight')
				plt.close()
			# video.write(overlaid)
		df_prob = pd.DataFrame(class_probs,columns=[path_col,class_col,'prob'])
		df_prob.to_csv(os.path.join(save_root,'class_probs.csv'), index=False)




	def overlay_cam(self, cam, raw_image, paper_cmap=False):
		"""
		Copied from save_gradcam in https://github.com/kazuto1011/grad-cam-pytorch/blob/master/main.py
		"""
		cmap = cm.jet(cam)[..., :3] * 255.0
		if paper_cmap:
			alpha = cam[..., None]
			cam = alpha * cmap + (1 - alpha) * raw_image
		else:
			cam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
		return cam.round().astype(np.uint8)


def get_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('model_weight', type=str, help='Path to the checkpoint of the pretrained model.')
	parser.add_argument('input_root', type=str, help='Path to the root annotationy under which inputs are located.')
	parser.add_argument('annotation_file', type=str, help='Path to the annotation csv.')
	parser.add_argument('save_root', type=str, help='Path to the directory where results are saved.')
	parser.add_argument('-d', '--device', type=str, default='cpu', help='Computing device.')
	# parser.add_argument('-b', '--batch_size', type=int, default=512, help='Batch size for training.')
	# parser.add_argument('--num_workers', type=int, default=1, help='# of workers.')
	parser.add_argument('--class_col', type=str, default='sex', help='Column in the annotation csv for the target label.')
	parser.add_argument('--path_col', type=str, default='stl_path', help='Column in the annotation csv for the relative path to the 3D mesh data.')
	parser.add_argument('--input_edge_size', type=int, default=224, help='The height and width of the resized bboxed image.')
	parser.add_argument('--elevs', type=float, nargs='+', default=[30.0, -30.0], help='Elevation degrees.')
	parser.add_argument('--num_azims', type=int, default=8, help='# of view azimuths.')
	# parser.add_argument('--azims', type=float, nargs='+', default=(np.arange(8) * 360.0 / 8).tolist(), help='Azimuth degrees.')
	parser.add_argument('--img_line_width', type=float, default=0.01, help='Line width of surface images.')
	parser.add_argument('--img_alpha', type=float, default=0.1, help='Opacity of surface images.')
	parser.add_argument('--grad_cam', action='store_true', help='If selected, Grad-CAM is used. Otherwise, Score-CAM is used.')
	parser.add_argument('--normalize_per_view', action='store_true', help='If selected, CAM weights are normalized within (not across) the views.')
	parser.add_argument('--rendering_engine', type=str, default='mpl', help='3D rendering engine to use. "mpl" (for matplotlib) or "pyvista".')

	return parser.parse_args()


if __name__ == '__main__':
	args = get_args()
	# Get a model.
	predictor = Predictor(args.model_weight,
				device = args.device, use_grad_cam=args.grad_cam,
				normalize_per_view=args.normalize_per_view
				)

	df_ann = pd.read_csv(args.annotation_file)
	df_ann[args.class_col] = df_ann[args.class_col].astype(str)

	class2ix_path = os.path.join(os.path.dirname(args.model_weight), 'target2ix.json')
	with open(class2ix_path, 'r') as f:
		class2ix = json.load(f)

	# Predict
	predictor.predict_dataset(
		df_ann,
		args.path_col,
		args.class_col,
		args.input_root,
		args.save_root,
		args.input_edge_size,
		class2ix,
		elevs=args.elevs,
		azims=(np.arange(args.num_azims)* 360.0 / args.num_azims).tolist(),
		img_line_width=args.img_line_width,
		img_alpha=args.img_alpha,
		rendering_engine=args.rendering_engine
		)