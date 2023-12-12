import math

import torch
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Identity, Parameter, init
import torch.nn.functional as F


class Attention(Module):
	"""
	Obtained from timm: github.com:rwightman/pytorch-image-models
	"""

	def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
		super().__init__()
		self.num_heads = num_heads
		head_dim = dim // self.num_heads
		self.scale = head_dim ** -0.5

		self.qkv = Linear(dim, dim * 3, bias=False)
		self.attn_drop = Dropout(attention_dropout)
		self.proj = Linear(dim, dim)
		self.proj_drop = Dropout(projection_dropout)

	def forward(self, x):
		B, N, C = x.shape
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]

		attn = (q @ k.transpose(-2, -1)) * self.scale
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)

		x = (attn @ v).transpose(1, 2).reshape(B, N, C)
		x = self.proj(x)
		x = self.proj_drop(x)
		return x


class MaskedAttention(Module):
	def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
		super().__init__()
		self.num_heads = num_heads
		head_dim = dim // self.num_heads
		self.scale = head_dim ** -0.5

		self.qkv = Linear(dim, dim * 3, bias=False)
		self.attn_drop = Dropout(attention_dropout)
		self.proj = Linear(dim, dim)
		self.proj_drop = Dropout(projection_dropout)

	def forward(self, x, mask=None):
		B, N1, N2, C = x.shape
		qkv = self.qkv(x).reshape(B, N1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]

		attn = (q @ k.transpose(-2, -1)) * self.scale

		if mask is not None:
			mask_value = -torch.finfo(attn.dtype).max
			assert mask.shape[-1] == attn.shape[-1], 'mask has incorrect dimensions'
			mask = mask[:, None, :] * mask[:, :, None]
			mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
			attn.masked_fill_(~mask, mask_value)

		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)

		x = (attn @ v).transpose(1, 2).reshape(B, N1, C)
		x = self.proj(x)
		x = self.proj_drop(x)
		return x


class TransformerEncoderLayer(Module):
	"""
	Inspired by torch.nn.TransformerEncoderLayer and timm.
	"""

	def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
				 attention_dropout=0.1, drop_path_rate=0.1):
		super(TransformerEncoderLayer, self).__init__()
		self.pre_norm = LayerNorm(d_model)
		self.self_attn = Attention(dim=d_model, num_heads=nhead,
								   attention_dropout=attention_dropout, projection_dropout=dropout)

		self.linear1 = Linear(d_model, dim_feedforward)
		self.dropout1 = Dropout(dropout)
		self.norm1 = LayerNorm(d_model)
		self.linear2 = Linear(dim_feedforward, d_model)
		self.dropout2 = Dropout(dropout)

		self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()

		self.activation = F.gelu

	def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
		src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
		src = self.norm1(src)
		src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
		src = src + self.drop_path(self.dropout2(src2))
		return src


class MaskedTransformerEncoderLayer(Module):
	"""
	Inspired by torch.nn.TransformerEncoderLayer and timm.
	"""

	def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
				 attention_dropout=0.1, drop_path_rate=0.1):
		super(MaskedTransformerEncoderLayer, self).__init__()
		self.pre_norm = LayerNorm(d_model)
		self.self_attn = MaskedAttention(dim=d_model, num_heads=nhead,
										 attention_dropout=attention_dropout, projection_dropout=dropout)

		self.linear1 = Linear(d_model, dim_feedforward)
		self.dropout1 = Dropout(dropout)
		self.norm1 = LayerNorm(d_model)
		self.linear2 = Linear(dim_feedforward, d_model)
		self.dropout2 = Dropout(dropout)

		self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()

		self.activation = F.gelu

	def forward(self, src: torch.Tensor, mask=None, *args, **kwargs) -> torch.Tensor:
		import pdb;pdb.set_trace()
		src = src + self.drop_path(self.self_attn(self.pre_norm(src), mask))
		src = self.norm1(src)
		src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
		src = src + self.drop_path(self.dropout2(src2))
		return src


class MaskedTransformerClassifier(Module):
	def __init__(self,
				 seq_pool=True,
				 embedding_dim=768,
				 num_layers=12,
				 num_heads=12,
				 mlp_ratio=4.0,
				 num_classes=1000,
				 dropout=0.1,
				 attention_dropout=0.1,
				 stochastic_depth=0.1,
				 positional_embedding='sine',
				 max_num_paragraphs=128,
				 max_paragraph_length=128,
				 *args, **kwargs):
		super().__init__()
		positional_embedding = positional_embedding if \
			positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
		dim_feedforward = int(embedding_dim * mlp_ratio)
		self.embedding_dim = embedding_dim
		self.max_num_paragraphs = max_num_paragraphs
		self.max_paragraph_length = max_paragraph_length
		self.seq_pool = seq_pool
		self.num_tokens = 0

		if not seq_pool:
			seq_len += 1
			self.class_emb = Parameter(torch.zeros(1, 1, self.embedding_dim),
									   requires_grad=True)
			self.num_tokens = 1
		else:
			self.attention_pool = Linear(self.embedding_dim, 1)

		if positional_embedding != 'none':
			if positional_embedding == 'learnable':
				seq_len += 1  # padding idx
				self.positional_emb = Parameter(torch.zeros(1, seq_len, embedding_dim),
												requires_grad=True)
				init.trunc_normal_(self.positional_emb, std=0.2)
			else:
				self.positional_emb = Parameter(
					self.sinusoidal_embedding(max_num_paragraphs, max_paragraph_length,										  
											  embedding_dim, padding_idx=True),
												requires_grad=False)
		else:
			self.positional_emb = None

		self.dropout = Dropout(p=dropout)
		dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
		self.blocks = ModuleList([
			MaskedTransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
										  dim_feedforward=dim_feedforward, dropout=dropout,
										  attention_dropout=attention_dropout, drop_path_rate=dpr[i])
			for i in range(num_layers)])
		self.norm = LayerNorm(embedding_dim)

		self.fc = Linear(embedding_dim, num_classes)
		self.apply(self.init_weight)

	def forward(self, x, mask=None):
		if self.positional_emb is None and x.size(1) < self.seq_len:
			x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

		if not self.seq_pool:
			cls_token = self.class_emb.expand(x.shape[0], -1, -1)
			x = torch.cat((cls_token, x), dim=1)
			if mask is not None:
				mask = torch.cat([torch.ones(size=(mask.shape[0], 1), device=mask.device), mask.float()], dim=1)
				mask = (mask > 0)

		if self.positional_emb is not None:
			x += self.positional_emb

		x = self.dropout(x)

		for blk in self.blocks:
			x = blk(x, mask=mask)
		x = self.norm(x)

		if self.seq_pool:
			x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
		else:
			x = x[:, 0]

		x = self.fc(x)
		return x

	@staticmethod
	def init_weight(m):
		if isinstance(m, Linear):
			init.trunc_normal_(m.weight, std=.02)
			if isinstance(m, Linear) and m.bias is not None:
				init.constant_(m.bias, 0)
		elif isinstance(m, LayerNorm):
			init.constant_(m.bias, 0)
			init.constant_(m.weight, 1.0)

	@staticmethod
	def sinusoidal_embedding(max_num_paragraphs, max_paragraph_length, embedding_dim, padding_idx=False):
		#pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / embedding_dim)) for i in range(embedding_dim)]
		#						for p in range(n_channels)])
		#pe[:, 0::2] = torch.sin(pe[:, 0::2])
		#pe[:, 1::2] = torch.cos(pe[:, 1::2])
		#print(pe.size())
		#pe = pe.unsqueeze(0)
		#print(pe.size())
		#print(padding_idx)
		#if padding_idx:
		#	return torch.cat([torch.zeros((1, 1, dim)), pe], dim=1)
		#return pe

		# From https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
		pe = torch.zeros(max_num_paragraphs, max_paragraph_length, embedding_dim)
	    # Each dimension use half of embedding_dim
		embedding_dim = int(embedding_dim / 2)
		div_term = torch.exp(torch.arange(0., embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))
		pos_l = torch.arange(0., max_paragraph_length).unsqueeze(1)
		pos_n = torch.arange(0., max_num_paragraphs).unsqueeze(1)
		pe[:, :, 0:embedding_dim:2] = torch.sin(pos_n * div_term).unsqueeze(0).repeat(max_num_paragraphs, 1, 1)
		pe[:, :, 1:embedding_dim:2] = torch.cos(pos_n * div_term).unsqueeze(0).repeat(max_num_paragraphs, 1, 1)
		pe[:, :, embedding_dim::2] = torch.sin(pos_l * div_term).unsqueeze(1).repeat(1, max_paragraph_length, 1)
		pe[:, :, embedding_dim + 1::2] = torch.cos(pos_l * div_term).unsqueeze(1).repeat(1, max_paragraph_length, 1)
		return pe

def drop_path(x, drop_prob: float = 0., training: bool = False):
	"""
	Obtained from: github.com:rwightman/pytorch-image-models
	Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
	This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
	the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
	See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
	changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
	'survival rate' as the argument.
	"""
	if drop_prob == 0. or not training:
		return x
	keep_prob = 1 - drop_prob
	shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
	random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
	random_tensor.floor_()  # binarize
	output = x.div(keep_prob) * random_tensor
	return output


class DropPath(Module):
	"""
	Obtained from: github.com:rwightman/pytorch-image-models
	Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
	"""

	def __init__(self, drop_prob=None):
		super(DropPath, self).__init__()
		self.drop_prob = drop_prob

	def forward(self, x):
		return drop_path(x, self.drop_prob, self.training)

