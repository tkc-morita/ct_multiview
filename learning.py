# encoding: utf-8

import torch
import torchvision.transforms
import numpy as np
from modules import model, data_utils, lr_scheduler
from logging import getLogger,FileHandler,DEBUG,Formatter
import os, argparse, itertools, json

logger = getLogger(__name__)

def update_log_handler(file_dir):
	current_handlers=logger.handlers[:]
	for h in current_handlers:
		logger.removeHandler(h)
	log_file_path = os.path.join(file_dir,'history.log')
	if os.path.isfile(log_file_path):
		retrieval = True
	else:
		retrieval = False
	handler = FileHandler(filename=log_file_path)	#Define the handler.
	handler.setLevel(DEBUG)
	formatter = Formatter('{asctime} - {levelname} - {message}', style='{')	#Define the log format.
	handler.setFormatter(formatter)
	logger.setLevel(DEBUG)
	logger.addHandler(handler)	#Register the handler for the logger.
	if retrieval:
		logger.info("LEARNING RETRIEVED.")
	else:
		logger.info("Logger set up.")
		logger.info("PyTorch ver.: {ver}".format(ver=torch.__version__))
	return retrieval,log_file_path



class Learner(object):
	def __init__(self,
			save_dir,
			num_categories,
			num_views,
			cnn='resnet50',
			aggregator='cnn',
			device='cpu',
			seed=1111,
			freeze_at=None,
			init_weight_path=None,
			transformer_kwargs=dict(),
			cnn_kwargs=dict()
			):
		self.retrieval,self.log_file_path = update_log_handler(save_dir)
		self.device = torch.device(device)
		logger.info('Device: {device}'.format(device=device))
		self.distributed = False
		if torch.cuda.is_available():
			if device.startswith('cuda'):
				logger.info('CUDA Version: {version}'.format(version=torch.version.cuda))
				if torch.backends.cudnn.enabled:
					logger.info('cuDNN Version: {version}'.format(version=torch.backends.cudnn.version()))
				# torch.distributed.init_process_group('nccl', rank=0, world_size=1) # Currently only support single-process with multi-gpu situation.
				self.distributed = True
			else:
				print('CUDA is available. Restart with option -C or --cuda to activate it.')



		self.save_dir = save_dir

		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

		self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='sum')

		if self.retrieval:
			logger.warning('The current implementation does not retrieve the random state of the dataloader workers.')
			self.last_iteration = self.retrieve_model(device=device)
			logger.info('Model retrieved.')
		else:
			torch.manual_seed(seed)
			torch.cuda.manual_seed_all(seed) # According to the docs, "It’s safe to call this function if CUDA is not available; in that case, it is silently ignored."
			self.seed = seed
			logger.info('Random seed: {seed}'.format(seed = seed))
			logger.info("# of categories: {num_categories}".format(num_categories=num_categories))
			
			if not init_weight_path is None:
				logger.info("Starts with the pretrained weights at {}".format(init_weight_path))
				self.retrieve_model(device=device, checkpoint_path=init_weight_path, only_load_model=True)
				self.model.freeze_layers(freeze_at)
			else:
				logger.info("View Encoder CNN: {cnn}".format(cnn=cnn))
				logger.info("Multiview aggregator: {}".format(aggregator))
				self.model = model.MultiViewClassifier(
								num_categories,
								num_views,
								cnn=cnn,
								aggregator=aggregator,
								in_channels=1,
								freeze_at=freeze_at,
								transformer_kwargs=transformer_kwargs,
								cnn_kwargs=cnn_kwargs,
								)
			if not freeze_at is None:
				logger.info("ResNet layers below layer{} are freezed.".format(freeze_at))
			self.model_init_args = self.model.pack_init_args()
			if self.distributed:
				self.model = torch.nn.DataParallel(self.model)
			self.model.to(self.device)


	def train(self, dataloader, saving_interval, start_iter=0, k=1):
		"""
		Training phase. Updates weights.
		"""
		self.model.train() # Turn on training mode which enables dropout.
		self.cross_entropy_loss.train()

		num_iterations = len(dataloader)
		total_loss = 0.0
		data_size = 0.0
		accuracy = 0.0

		for iteration,(views, classes, _, _) in enumerate(dataloader, start_iter):
			iteration += 1 # Original starts with 0.
			

			views = [v.to(self.device) for v in views]
			classes = classes.to(self.device)

			self.optimizer.zero_grad()
			torch.manual_seed(iteration+self.seed)
			torch.cuda.manual_seed_all(iteration+self.seed)

			# with torch.autograd.detect_anomaly():
			logits = self.model(views)
			loss = self.cross_entropy_loss(logits, classes)
			(loss / views[0].size(0)).backward()

			self.optimizer.step()
			self.lr_scheduler.step()

			total_loss += loss.item()
			data_size += views[0].size(0)
			accuracy += (logits.topk(k,-1).indices==classes.view(-1,1)).any(-1).float().sum().item()

			if iteration % saving_interval == 0:
				mean_loss = total_loss / data_size
				perplexity = np.exp(mean_loss)
				accuracy = accuracy / data_size
				logger.info('{iteration}/{num_iterations} iterations complete. mean loss (perplexity): {loss} ({perplexity}). top-{k} accuracy: {accuracy}.'.format(iteration=iteration, num_iterations=num_iterations, loss=mean_loss, perplexity=perplexity, accuracy=accuracy, k=k))
				total_loss = 0.0
				data_size = 0.0
				accuracy = 0.0
				self.save_model(iteration-1)
		self.save_model(iteration-1)



	def learn(self, train_dataset, num_iterations, batch_size_train, milestones, num_workers=1, learning_rate=0.1, momentum= 0.9, decay=5.0*(10**-4), gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=0, warmup_method='linear', saving_interval=200):
		if self.retrieval:
			start_iter = self.last_iteration + 1
			logger.info('To be restarted from the beginning of iteration #: {iteration}'.format(iteration=start_iter+1))
			self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)
			self.optimizer.load_state_dict(self.checkpoint['optimizer'])

			self.lr_scheduler = lr_scheduler.WarmupMultiStepLR(self.optimizer, milestones, gamma=gamma, warmup_factor=warmup_factor, warmup_iters=warmup_iters, warmup_method=warmup_method)
			self.lr_scheduler.load_state_dict(self.checkpoint['lr_scheduler'])
		else:
			self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)
			self.lr_scheduler = lr_scheduler.WarmupMultiStepLR(self.optimizer, milestones, gamma=gamma, warmup_factor=warmup_factor, warmup_iters=warmup_iters, warmup_method=warmup_method)
			logger.info("START LEARNING.")
			logger.info("max # of iterations: {ep}".format(ep=num_iterations))
			logger.info("batch size for training data: {size}".format(size=batch_size_train))
			logger.info("initial learning rate: {lr}".format(lr=learning_rate))
			logger.info("momentum for SGD: {momentum}".format(momentum=momentum))
			logger.info("weight decay: {decay}".format(decay=decay))
			logger.info("First {warmup_iters} iterations for warm-up.".format(warmup_iters=warmup_iters))
			logger.info("warmup_factor: {warmup_factor}".format(warmup_factor=warmup_factor))
			logger.info("warmup_method: {warmup_method}".format(warmup_method=warmup_method))
			logger.info("milestones: {milestones}".format(milestones=milestones))
			logger.info("gamma: {gamma}".format(gamma=gamma))
			start_iter = 0
		train_dataloader = data_utils.make_data_loader(
			train_dataset,
			batch_size_train,
			is_train=True,
			start_iter=start_iter, 
			num_workers=num_workers,
			num_iters=num_iterations,
			random_seed=self.seed
			)
		self.train(train_dataloader, saving_interval, start_iter=start_iter)
		logger.info('END OF TRAINING')


	def save_model(self, iteration):
		"""
		Save model config.
		Allow multiple tries to prevent immediate I/O errors.
		"""
		checkpoint = {
			'iteration':iteration,
			'model':self.model.state_dict(),
			'model_init_args':self.model_init_args,
			'optimizer':self.optimizer.state_dict(),
			'lr_scheduler':self.lr_scheduler.state_dict(),
			'distributed':self.distributed,
			'random_seed':self.seed,
		}
		if torch.cuda.is_available():
			checkpoint['random_state_cuda'] = torch.cuda.get_rng_state_all()
		torch.save(checkpoint, os.path.join(self.save_dir, 'checkpoint_after-{iteration}-iters.pt'.format(iteration=iteration+1)))
		torch.save(checkpoint, os.path.join(self.save_dir, 'checkpoint.pt'))
		logger.info('Config successfully saved.')


	def retrieve_model(self, checkpoint_path = None, device='cpu', only_load_model=False, no_parallel=False):
		if checkpoint_path is None:
			checkpoint_path = os.path.join(self.save_dir, 'checkpoint.pt')
		self.checkpoint = torch.load(checkpoint_path, map_location='cpu') # Random state needs to be loaded to CPU first even when cuda is available.


		self.model = model.MultiViewClassifier(**self.checkpoint['model_init_args'])
		model_state_dict = {('^'+key).replace('^module.', '').replace('^',''):value
							for key,value in self.checkpoint['model'].items()}
		self.model.load_state_dict(model_state_dict)
		if not only_load_model:
			self.model_init_args = self.model.pack_init_args()
			if self.checkpoint['distributed'] and not no_parallel:
				self.model = torch.nn.DataParallel(self.model)
			self.model.to(self.device) # This needs to be above self.optimizer.load_state_dict(). See https://discuss.pytorch.org/t/runtimeerror-expected-type-torch-floattensor-but-got-torch-cuda-floattensor-while-resuming-training/37936/5.
			
			self.seed = self.checkpoint['random_seed']
			return self.checkpoint['iteration']



def get_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('train_input_root', type=str, help='Path to the root annotationy under which inputs are located.')
	parser.add_argument('train_annotation_file', type=str, help='Path to the annotation csv file for the training data.')
	parser.add_argument('class_col', type=str, help='Column of train_annotation_file encoding the classification.')
	parser.add_argument('--path_col', type=str, default='stl_path', help='Column of train_annotation_file encoding the path to the STL files relative to train_input_root.')
	parser.add_argument('-S', '--save_root', type=str, default=None, help='Path to the annotationy where results are saved.')
	parser.add_argument('--input_edge_size', type=int, default=224, help='The height and width of the resized bboxed image.')
	parser.add_argument('-j', '--job_id', type=str, default='NO_JOB_ID', help='Job ID. For users of computing clusters.')
	parser.add_argument('-s', '--seed', type=int, default=1111, help='random seed')
	parser.add_argument('-d', '--device', type=str, default='cpu', help='Computing device.')
	parser.add_argument('--cnn', type=str, default='resnet50', help='Name of the CNN for feature extraction. Choose one from torchvision.models.')
	parser.add_argument('-i', '--iterations', type=int, default=25000, help='# of iterations to train the model.')
	parser.add_argument('-b', '--batch_size', type=int, default=512, help='Batch size for training.')
	parser.add_argument('--learning_rate', type=float, default=0.1, help='Initial learning rate.')
	parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for the storchastic gradient descent.')
	parser.add_argument('--decay', type=float, default=5.0*(10**-4), help='Weight decay.')
	parser.add_argument('--milestones', type=int, nargs='+', default=[20000,28000], help='Milestones at which the learning rate is updated.')
	parser.add_argument('--gamma', type=int, default=0.1, help='Factor of the update on the learning rate.')
	parser.add_argument('--warmup_factor', type=float, default=1.0 / 3, help='')
	parser.add_argument('--warmup_iters', type=int, default=0, help='# of iterations for warmup.')
	parser.add_argument('--warmup_method', type=str, default='linear', help='linear or constant warmup.')
	parser.add_argument('--num_workers', type=int, default=1, help='# of workers (>=1).')
	parser.add_argument('--saving_interval', type=int, default=500, help='# of iterations in which model parameters are saved once.')
	parser.add_argument('--freeze_at', type=int, default=None, help='ResNet layers below this level are freezed. If unspecified, train all including the bottom convolution.')
	parser.add_argument('--weight', type=str, default=None, help='Path to the checkpoint of the pretrained model.')
	parser.add_argument('--img_line_width', type=float, default=0.01, help='Line width of surface images.')
	parser.add_argument('--img_alpha', type=float, default=0.1, help='Opacity of surface images.')
	parser.add_argument('--elevs', type=float, nargs='+', default=[30.0, -30.0], help='Elevation of the views [-90,90].')
	parser.add_argument('--num_azims', type=int, default=8, help='# of view azimuths.')
	parser.add_argument('--noise_range', type=float, default=10.0, help='Range of noise on the azimuths [0,180).')
	parser.add_argument('--aggregator', type=str, default='cnn', help='Type of view aggregator. Either "cnn" or "transformer".')
	parser.add_argument('--transformer_dim', type=int, default=256, help='Dimensionality of the transformer aggregator.')
	parser.add_argument('--transformer_layers', type=int, default=3, help='# of layers of the transformer aggregator.')
	parser.add_argument('--transformer_heads', type=int, default=4, help='# of heads of the transformer aggregator.')
	parser.add_argument('--transformer_ff_dim', type=int, default=256, help='Dimensionality of the feed-forward layer of the transformer aggregator.')
	parser.add_argument('--rendering_engine', type=str, default='mpl', help='3D rendering engine to use. "mpl" (for matplotlib), "pyvista", "pyrender", or "plotly".')

	return parser.parse_args()


def get_save_dir(save_root, job_id_str):
	save_dir = os.path.join(
					save_root,
					job_id_str # + '_START-AT-' + datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S-%f')
				)
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	return save_dir

if __name__ == '__main__':
	args = get_args()

	save_root = args.save_root
	if save_root is None:
		save_root = args.input_root
	save_dir = get_save_dir(save_root, args.job_id)


	transforms = [
		# torchvision.transforms.Resize((args.input_height, args.input_width)),
		data_utils.ProportionalResizeAndPad(args.input_edge_size),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5,), (0.5,))
	]

	target2ix_path = os.path.join(save_dir, 'target2ix.json')
	if os.path.isfile(target2ix_path):
		with open(target2ix_path, 'r') as f:
			target2ix = json.load(f)
	else:
		target2ix = None

	# if args.rendering_engine=='pyvista' and os.uname().sysname!='Darwin':
	# 	from pyvirtualdisplay import Display
	# 	display = Display(visible=0, size=(512, 512))
	# 	display.start()
	# else:
	# 	display = None

	train_dataset = data_utils.ThreeDSurfaceDataset(
						ann_file=args.train_annotation_file,
						root=args.train_input_root,
						target_col=args.class_col,
						path_col=args.path_col,
						target2ix=target2ix,
						transforms=torchvision.transforms.Compose(transforms),
						img_line_width=args.img_line_width,
						img_alpha=args.img_alpha,
						elevs=args.elevs,
						num_azims=args.num_azims,
						noise_range=args.noise_range,
						rendering_engine=args.rendering_engine
						)

	if target2ix is None:
		with open(target2ix_path, 'w') as f:
			json.dump(train_dataset.target2ix, f)

	num_categories = train_dataset.get_num_categories()

	num_views = args.num_azims*len(args.elevs)
	# Get a model.
	learner = Learner(
				save_dir,
				num_views=num_views,
				device = args.device,
				seed = args.seed,
				cnn=args.cnn,
				num_categories=num_categories,
				freeze_at=args.freeze_at,
				init_weight_path=args.weight,
				aggregator=args.aggregator,
				transformer_kwargs={
					"d_model":args.transformer_dim,
					"nheads":args.transformer_heads,
					"num_layers":args.transformer_layers,
					"dim_feedforward":args.transformer_ff_dim,
				},
				)

	logger.info('Classification based on {} views.'.format(num_views))
	logger.info('View elevations: {}'.format(args.elevs))
	logger.info('Center of view azimuths: {}'.format(train_dataset.azims_center.tolist()))
	half_noise = args.noise_range * 0.5
	logger.info('Azimuth noise range: [{}, {}]'.format(-half_noise,half_noise))
	logger.info('Rendering engine: {}'.format(args.rendering_engine))
	if args.rendering_engine=='mpl':
		logger.info('Line width of surface images: {}'.format(args.img_line_width))
		logger.info('Opacity (alpha) of surface images: {}'.format(args.img_alpha))

	assert args.num_workers>0, '--num_workers must be a positive integer.'
	# Train the model.
	learner.learn(
			train_dataset,
			args.iterations,
			args.batch_size,
			args.milestones,
			learning_rate=args.learning_rate,
			momentum=args.momentum,
			decay = args.decay,
			gamma=args.gamma,
			warmup_factor=args.warmup_factor,
			warmup_iters=args.warmup_iters,
			warmup_method=args.warmup_method,
			num_workers=args.num_workers,
			saving_interval=args.saving_interval
			)

	# if not display is None:
	# 	display.stop()