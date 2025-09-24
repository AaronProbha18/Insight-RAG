from sentence_transformers import SentenceTransformer

def get_embedding_model(model_name='all-mpnet-base-v2'):
	return SentenceTransformer(model_name)

def generate_embeddings(model, text_chunks):
	return model.encode(text_chunks)
