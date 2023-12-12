# This model is based on the compact transformer model (text_cct_2) described at
# https://medium.com/pytorch/training-compact-transformers-from-scratch-in-30-minutes-with-pytorch-ff5c21668ed5

import torch.nn as nn
from .transformer import MaskedTransformerClassifier
from .tokenizer import Tokenizer

EMBEDDING_SIZE = 100
MAX_PARAGRAPH_LENGTH = 128
MAX_NUM_PARAGRAPHS = 128

def primary_model(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128,
			  kernel_size=4, stride=None, padding=None,
			  *args, **kwargs):
	stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
	padding = padding if padding is not None else max(1, (kernel_size // 2))

	return TextCCT(num_layers=2,
				   num_heads=2,
				   mlp_ratio=1,
				   embedding_dim=EMBEDDING_SIZE,
				   kernel_size=4,
				   stride=1,
				   padding=1,
				   *args, **kwargs)

class TextCCT(nn.Module):
	def __init__(self,
				 seq_len=64,
				 word_embedding_dim=300,
				 embedding_dim=128,
				 kernel_size=2,
				 stride=1,
				 padding=1,
				 pooling_kernel_size=2,
				 pooling_stride=2,
				 pooling_padding=1,
				 *args, **kwargs):
		super(TextCCT, self).__init__()

		self.classifier = MaskedTransformerClassifier(
			seq_len=MAX_NUM_PARAGRAPHS,
			embedding_dim=embedding_dim,
			dropout=0.,
			attention_dropout=0.1,
			stochastic_depth=0.1,
			*args, **kwargs)
		
	def forward(self, x, mask=None):
		out = self.classifier(x, mask=mask)
		return out
