import torch

class SimpleVectorStore:
	def __init__(self, embeddings, texts, page_numbers, sources):
		self.embeddings = embeddings
		self.texts = texts
		self.page_numbers = page_numbers
		self.sources = sources

	def search(self, query_embedding, top_k=3):
		similarities = torch.nn.functional.cosine_similarity(
			torch.tensor(query_embedding).unsqueeze(0),
			torch.tensor(self.embeddings)
		)
		top_indices = similarities.argsort(descending=True)[:top_k]
		return [(self.texts[i], self.page_numbers[i], self.sources[i]) for i in top_indices]
