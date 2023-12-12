#!/usr/bin/env python

import argparse
from time import time
import math

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

from src_model.embeddings import Embeddings
from src_model.model import primary_model
from src_model.loss import LabelSmoothingCrossEntropy

from src_preprocessor.process_text import clean_text
from src_preprocessor.text_utils import to_paragraphs

from torch.utils.data import Dataset, DataLoader
import pandas

best_acc1 = 0

DATASET_CSV = 'dataset.csv'
GLOVE_DATASET_FILE = '../glove.6B/glove.6B.100d.txt'

EMBEDDING_SIZE = 100
MAX_PARAGRAPH_LENGTH = 128
MAX_NUM_PARAGRAPHS = 128

def get_dataset(Embeddings, start=0, end=-1):
	dataset = CustomDataset(DATASET_CSV, start, end, Embeddings)
	return dataset


class CustomDataset(Dataset):
	def __init__(self, csv_file, start, end, Embeddings):
		self.start = start
		self.end = end
		self.dataframe = pandas.read_csv(csv_file)
		self.embeddings = Embeddings
		if end < 0:
			self.len = len(self.dataframe) - start
		else:
			self.len = end - start

	def __len__(self):
		return self.len

	def __getitem__(self, i):
		i += self.start
		file_id = self.dataframe.loc[i, 'file_id']
		para_n = torch.tensor(self.dataframe.loc[i, 'para_n'], dtype=torch.int)
		
		with open(f'casefiles_raw/{file_id}', 'r') as file:
			text = file.read()
		text = to_paragraphs(text)
		if len(text) > MAX_NUM_PARAGRAPHS:
			text = text[:MAX_NUM_PARAGRAPHS]
		text = [clean_text(paragraph) for paragraph in text]
		text = [self.embeddings.to_embeddings_paragraph(paragraph) for paragraph in text]
		while len(text) < MAX_NUM_PARAGRAPHS:
			text.append(torch.zeros_like(text[0]))
		text = torch.stack(text)
		text = text.detach()

		return text, para_n


def collate_fn(batch):
	text = torch.stack([item[0] for item in batch])
	para_n = torch.stack([item[1] for item in batch])
	return text, para_n


def init_parser():
	parser = argparse.ArgumentParser(description='CIFAR quick training script')

	# Data args
	parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
						help='number of data loading workers (default: 4)')

	parser.add_argument('--print-freq', default=10, type=int, metavar='N',
						help='log frequency (by iteration)')

	parser.add_argument('--checkpoint-path',
						type=str,
						default='checkpoint.pth',
						help='path to checkpoint (default: checkpoint.pth)')

	# Optimization hyperparams
	parser.add_argument('--epochs', default=200, type=int, metavar='N',
						help='number of total epochs to run')
	parser.add_argument('--warmup', default=5, type=int, metavar='N',
						help='number of warmup epochs')
	parser.add_argument('-b', '--batch-size', default=128, type=int,
						metavar='N',
						help='mini-batch size (default: 128)', dest='batch_size')
	parser.add_argument('--lr', default=0.0005, type=float,
						help='initial learning rate')
	parser.add_argument('--weight-decay', default=3e-2, type=float,
						help='weight decay (default: 1e-4)')
	parser.add_argument('--clip-grad-norm', default=0., type=float,
						help='gradient norm clipping (default: 0 (disabled))')

	parser.add_argument('-p', '--positional-embedding',
						type=str.lower,
						choices=['learnable', 'sine', 'none'],
						default='learnable', dest='positional_embedding')

	parser.add_argument('--conv-layers', default=2, type=int,
						help='number of convolutional layers (cct only)')

	parser.add_argument('--conv-size', default=3, type=int,
						help='convolution kernel size (cct only)')

	parser.add_argument('--patch-size', default=4, type=int,
						help='image patch size (vit and cvt only)')

	parser.add_argument('--disable-cos', action='store_true',
						help='disable cosine lr schedule')

	parser.add_argument('--disable-aug', action='store_true',
						help='disable augmentation policies for training')

	parser.add_argument('--gpu-id', default=0, type=int)

	parser.add_argument('--no-cuda', action='store_true',
						help='disable cuda')

	return parser


def main():
	global best_acc1

	parser = init_parser()
	args = parser.parse_args()

	model = primary_model()

	criterion = LabelSmoothingCrossEntropy()

	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
								  weight_decay=args.weight_decay)

	embeddings = Embeddings(GLOVE_DATASET_FILE)
	train_dataset = get_dataset(embeddings, start=10)
	val_dataset = get_dataset(embeddings, end=10)

	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn,
		num_workers=args.workers)

	val_loader = torch.utils.data.DataLoader(
		val_dataset,
		batch_size=1, shuffle=False, collate_fn=collate_fn,
		num_workers=args.workers)

	print("Beginning training")
	time_begin = time()
	for epoch in range(args.epochs):
		adjust_learning_rate(optimizer, epoch, args)
		cls_train(train_loader, model, criterion, optimizer, epoch, args)
		acc1 = cls_validate(val_loader, model, criterion, args, epoch=epoch, time_begin=time_begin)
		best_acc1 = max(acc1, best_acc1)

	total_mins = (time() - time_begin) / 60
	print(f'Script finished in {total_mins:.2f} minutes, '
		  f'best top-1: {best_acc1:.2f}, '
		  f'final top-1: {acc1:.2f}')
	torch.save(model.state_dict(), args.checkpoint_path)


def adjust_learning_rate(optimizer, epoch, args):
	lr = args.lr
	if hasattr(args, 'warmup') and epoch < args.warmup:
		lr = lr / (args.warmup - epoch)
	elif not args.disable_cos:
		lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup) / (args.epochs - args.warmup)))

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def accuracy(output, target):
	with torch.no_grad():
		batch_size = target.size(0)

		_, pred = output.topk(1, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		correct_k = correct[:1].flatten().float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
		return res



def cls_train(train_loader, model, criterion, optimizer, epoch, args):
	model.train()
	loss_val, acc1_val = 0, 0
	n = 0
	for i, (text, target) in enumerate(train_loader):
		output = model(text)

		loss = criterion(output, target)

		acc1 = accuracy(output, target)
		n += text.size(0)
		loss_val += float(loss.item() * text.size(0))
		acc1_val += float(acc1[0] * text.size(0))

		optimizer.zero_grad()
		loss.backward()

		if args.clip_grad_norm > 0:
			nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm, norm_type=2)

		optimizer.step()

		if args.print_freq >= 0 and i % args.print_freq == 0:
			avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
			print(f'[Epoch {epoch + 1}][Train][{i}] \t Loss: {avg_loss:.4e} \t Top-1 {avg_acc1:6.2f}')


def cls_validate(val_loader, model, criterion, args, epoch=None, time_begin=None):
	model.eval()
	loss_val, acc1_val = 0, 0
	n = 0
	with torch.no_grad():
		for i, (images, target) in enumerate(val_loader):
			if (not args.no_cuda) and torch.cuda.is_available():
				images = images.cuda(args.gpu_id, non_blocking=True)
				target = target.cuda(args.gpu_id, non_blocking=True)

			output = model(images)
			loss = criterion(output, target)

			acc1 = accuracy(output, target)
			n += images.size(0)
			loss_val += float(loss.item() * images.size(0))
			acc1_val += float(acc1[0] * images.size(0))

			if args.print_freq >= 0 and i % args.print_freq == 0:
				avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
				print(f'[Epoch {epoch + 1}][Eval][{i}] \t Loss: {avg_loss:.4e} \t Top-1 {avg_acc1:6.2f}')

	avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
	total_mins = -1 if time_begin is None else (time() - time_begin) / 60
	print(f'[Epoch {epoch + 1}] \t \t Top-1 {avg_acc1:6.2f} \t \t Time: {total_mins:.2f}')

	return avg_acc1


if __name__ == '__main__':
	main()
