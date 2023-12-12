import torch

def to_paragraphs(text):
	return text.split('\n\n')
