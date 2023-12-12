# Utility to load GLoVe embeddings

import numpy
import torch
import torch.nn as nn

EMBEDDING_SIZE = 100
MAX_PARAGRAPH_LENGTH = 128

class Embeddings():
	def __init__(self, path):
		embeddings = {}
		with open(path, 'r', encoding='utf-8') as f:
			for line in f:
				values = line.split()
				word = values[0]
				vector = numpy.asarray(values[1:], "float32")
				tensor = torch.from_numpy(vector)
				embeddings[word] = tensor

		words = list(embeddings.keys())
		embedding_matrix = numpy.zeros((len(words), EMBEDDING_SIZE))
		self.word_to_index = {}
		for i, word in enumerate(words):
			embedding_matrix[i] = embeddings[word]
			self.word_to_index[word] = i

		self.embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
		self.embedding_matrix = self.embedding_matrix.detach()
		self.num_embeddings, self.embedding_dim = self.embedding_matrix.size()

		self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
		self.embedding.weight = nn.Parameter(self.embedding_matrix)
	
	def to_embeddings_paragraph(self, paragraph):
		words = paragraph.split()
		embeddings = []
		for word in words:
			index = self.to_index(word)
			if index is not None:
				embeddings.append(self.embedding(index))
			else:
				embeddings.append(torch.zeros(EMBEDDING_SIZE))
		while len(embeddings) < MAX_PARAGRAPH_LENGTH:
			embeddings.append(torch.zeros(EMBEDDING_SIZE))
		if len(embeddings) > MAX_PARAGRAPH_LENGTH:
			embeddings = embeddings[:MAX_PARAGRAPH_LENGTH]
		return torch.stack(embeddings)
	
	def to_index(self, word):
		word = word.lower()
		if word in self.word_to_index:
			return torch.tensor(self.word_to_index[word], dtype=torch.int32)
		return None
