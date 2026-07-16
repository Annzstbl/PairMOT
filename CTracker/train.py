import time
import os
random_seed = 20200804
os.environ['PYTHONHASHSEED'] = str(random_seed)
import copy
import argparse
import pdb
import collections
import sys
import random
random.seed(random_seed)
import numpy as np
np.random.seed(random_seed)

import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(random_seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed(random_seed)
	torch.cuda.manual_seed_all(random_seed)
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import torchvision

import model
from anchors import Anchors
from test import run_from_train
import losses
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer, PhotometricDistort, RandomSampleCrop
from hsmot_adapter import (HSMOT_CLASSES, build_hsmot_pair_dataset,
                           ctracker_collate)
from torch.utils.data import Dataset, DataLoader

print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(args=None):

	parser     = argparse.ArgumentParser(description='Simple training script for training a CTracker network.')

	parser.add_argument('--dataset', default='csv', choices=('csv', 'hsmot'))
	parser.add_argument('--model_dir', default='./ctracker/', type=str, help='Path to save the model.')
	parser.add_argument('--root_path', default='/dockerdata/home/changanwang/Dataset/Tracking/MOT17Det/', type=str, help='Path of the directory containing both label and images')
	parser.add_argument('--csv_train', default='train_annots.csv', type=str, help='Path to file containing training annotations (see readme)')
	parser.add_argument('--csv_classes', default='train_labels.csv', type=str, help='Path to file containing class list (see readme)')
	
	parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
	parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
	parser.add_argument('--batch_size', type=int, default=8)
	parser.add_argument('--workers', type=int, default=8)
	parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
	parser.add_argument('--no_pretrained', action='store_true', help='Do not load ImageNet pretrained weights.')
	parser.add_argument('--pretrained_model', default='',
	                    help='Original CTracker final.pt to adapt for HSMOT.')
	parser.add_argument('--lr', type=float, default=5e-5)
	parser.add_argument('--stem_lr_multiplier', type=float, default=1.0)
	parser.add_argument('--data_parallel', action='store_true', help='Use all visible CUDA devices.')
	parser.add_argument('--skip_test', action='store_true', help='Do not run MOT17 evaluation after training.')
	parser.add_argument('--ann_file', default='', help='Optional HSMOT sequence split file.')
	parser.add_argument('--ann_subdir', default='mot')
	parser.add_argument('--img_subdir', default='npy2jpg')
	parser.add_argument('--img_format', choices=('npy', '3jpg'), default='3jpg')
	parser.add_argument('--image_scale', type=int, nargs=2,
	                    default=(900, 1200), metavar=('H', 'W'))
	parser.add_argument('--no_augment', action='store_true')
	parser.add_argument('--max_iters', type=int, default=0,
	                    help='Stop after this many iterations; 0 means unlimited.')
	parser.add_argument('--resume', default='', help='Training checkpoint to resume.')
	parser.add_argument('--checkpoint_interval', type=int, default=1)

	parser = parser.parse_args(args)
	print(parser)
	device = torch.device(parser.device)
	if device.type == 'cuda' and not torch.cuda.is_available():
		raise RuntimeError('CUDA was requested but is not available.')
	
	print(parser.model_dir)
	if not os.path.exists(parser.model_dir):
	   os.makedirs(parser.model_dir)

	# Create the data loaders
	if parser.dataset == 'csv':
		if (parser.csv_train is None) or (parser.csv_train == ''):
			raise ValueError('Must provide --csv_train when training on COCO,')

		if (parser.csv_classes is None) or (parser.csv_classes == ''):
			raise ValueError('Must provide --csv_classes when training on COCO,')

		dataset_train = CSVDataset(parser.root_path, train_file=os.path.join(parser.root_path, parser.csv_train), class_list=os.path.join(parser.root_path, parser.csv_classes), \
			transform=transforms.Compose([RandomSampleCrop(), PhotometricDistort(), Augmenter(), Normalizer()]))#transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

		sampler = AspectRatioBasedSampler(
			dataset_train, batch_size=parser.batch_size, drop_last=False)
		dataloader_train = DataLoader(
			dataset_train, num_workers=parser.workers,
			collate_fn=collater, batch_sampler=sampler)
		num_classes = dataset_train.num_classes()
		model_kwargs = {}
	else:
		dataset_train = build_hsmot_pair_dataset(
			parser.root_path, ann_file=parser.ann_file,
			ann_subdir=parser.ann_subdir, img_subdir=parser.img_subdir,
			img_format=parser.img_format, training=True,
			image_scale=tuple(parser.image_scale),
			augment=not parser.no_augment)
		dataloader_train = DataLoader(
			dataset_train, batch_size=parser.batch_size, shuffle=True,
			num_workers=parser.workers, collate_fn=ctracker_collate,
			persistent_workers=False)
		num_classes = len(HSMOT_CLASSES)
		model_kwargs = dict(
			num_spectral=8, use_3d_se_stem=True, rotated=True)

	# Create the model
	use_imagenet_pretrain = not parser.no_pretrained and not parser.pretrained_model
	if parser.depth == 18:
		retinanet = model.resnet18(num_classes=num_classes, pretrained=use_imagenet_pretrain, **model_kwargs)
	elif parser.depth == 34:
		retinanet = model.resnet34(num_classes=num_classes, pretrained=use_imagenet_pretrain, **model_kwargs)
	elif parser.depth == 50:
		retinanet = model.resnet50(num_classes=num_classes, pretrained=use_imagenet_pretrain, **model_kwargs)
	elif parser.depth == 101:
		retinanet = model.resnet101(num_classes=num_classes, pretrained=use_imagenet_pretrain, **model_kwargs)
	elif parser.depth == 152:
		retinanet = model.resnet152(num_classes=num_classes, pretrained=use_imagenet_pretrain, **model_kwargs)
	else:
		raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')		

	if parser.pretrained_model:
		load_report = model.load_legacy_ctracker(
			retinanet, parser.pretrained_model)
		print('Loaded legacy CTracker weights: {}'.format(load_report))
	retinanet = retinanet.to(device)
	if parser.data_parallel:
		if device.type != 'cuda':
			raise ValueError('--data_parallel requires a CUDA device.')
		retinanet = torch.nn.DataParallel(retinanet)
	model_without_wrapper = retinanet.module if isinstance(retinanet, torch.nn.DataParallel) else retinanet

	retinanet.training = True

	stem_parameters = [
		parameter for parameter in model_without_wrapper.conv1.parameters()
		if parameter.requires_grad]
	stem_parameter_ids = {id(parameter) for parameter in stem_parameters}
	base_parameters = [
		parameter for parameter in model_without_wrapper.parameters()
		if parameter.requires_grad and id(parameter) not in stem_parameter_ids]
	optimizer = optim.Adam([
		dict(params=base_parameters, lr=parser.lr),
		dict(params=stem_parameters,
		     lr=parser.lr * parser.stem_lr_multiplier),
	], lr=parser.lr)
	print('Optimizer learning rates: base={}, stem={} ({}x)'.format(
		parser.lr, parser.lr * parser.stem_lr_multiplier,
		parser.stem_lr_multiplier))

	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
	start_epoch = 0
	total_iter = 0
	if parser.resume:
		checkpoint = torch.load(parser.resume, map_location=device)
		model_without_wrapper.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		if 'scheduler' in checkpoint:
			scheduler.load_state_dict(checkpoint['scheduler'])
		start_epoch = int(checkpoint.get('epoch', -1)) + 1
		total_iter = int(checkpoint.get('total_iter', 0))
		print('Resumed {} at epoch {}, iteration {}'.format(
			parser.resume, start_epoch, total_iter))

	loss_hist = collections.deque(maxlen=500)

	retinanet.train()
	model_without_wrapper.freeze_bn()

	print('Num training images: {}'.format(len(dataset_train)))
	for epoch_num in range(start_epoch, parser.epochs):
		if hasattr(dataset_train, 'set_epoch'):
			dataset_train.set_epoch(epoch_num)

		retinanet.train()
		model_without_wrapper.freeze_bn()
		
		epoch_loss = []
		
		for iter_num, data in enumerate(dataloader_train):
			try:
				total_iter = total_iter + 1
				optimizer.zero_grad()


				if parser.dataset == 'hsmot':
					model_inputs = dict(
						img_prev=data['img_prev'].to(device),
						img_curr=data['img_curr'].to(device),
						targets=data['targets'])
					loss_dict = {
						name: value.mean()
						for name, value in retinanet(model_inputs).items()
					}
					loss = sum(loss_dict.values())
				else:
					(classification_loss, regression_loss), reid_loss = retinanet([
						data['img'].to(device=device, dtype=torch.float32), data['annot'],
						data['img_next'].to(device=device, dtype=torch.float32), data['annot_next']
					])
					classification_loss = classification_loss.mean()
					regression_loss = regression_loss.mean()
					reid_loss = reid_loss.mean()
					loss_dict = dict(
						loss_cls=classification_loss,
						loss_delta=regression_loss,
						loss_assoc=reid_loss)
					loss = sum(loss_dict.values())
				
				if bool(loss == 0):
					continue

				loss.backward()

				torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

				optimizer.step()

				loss_hist.append(float(loss))
				epoch_loss.append(float(loss))

				loss_text = ' | '.join(
					'{}: {:.5f}'.format(name, float(value.detach()))
					for name, value in loss_dict.items())
				print('Epoch: {} | Iter: {} | {} | Running loss: {:.5f}'.format(
					epoch_num, iter_num, loss_text, np.mean(loss_hist)))
			except Exception as e:
				print(e)
				if parser.dataset == 'hsmot':
					raise
				continue

			if parser.max_iters and total_iter >= parser.max_iters:
				break

		if epoch_loss:
			scheduler.step(np.mean(epoch_loss))
		checkpoint = dict(
			epoch=epoch_num,
			total_iter=total_iter,
			model=model_without_wrapper.state_dict(),
			optimizer=optimizer.state_dict(),
			scheduler=scheduler.state_dict(),
			args=vars(parser))
		torch.save(
			checkpoint, os.path.join(parser.model_dir, 'checkpoint_latest.pt'))
		if (parser.checkpoint_interval > 0 and
				(epoch_num + 1) % parser.checkpoint_interval == 0):
			torch.save(checkpoint, os.path.join(
				parser.model_dir,
				'checkpoint_epoch_{:03d}.pt'.format(epoch_num + 1)))
		if parser.max_iters and total_iter >= parser.max_iters:
			break

	retinanet.eval()

	torch.save(model_without_wrapper, os.path.join(parser.model_dir, 'model_final.pt'))
	if not parser.skip_test and parser.dataset == 'csv':
		run_from_train(parser.model_dir, parser.root_path, device=device)

if __name__ == '__main__':
	main()
