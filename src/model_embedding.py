from sentence_transformers import SentenceTransformer  # Importing the SentenceTransformer class
from config import MODEL_NAME  # Configuration setting for the pre-trained model name
import random
import numpy as np

# Class for generating BERT embeddings using the SentenceTransformer library
class BlackModelEmbed():

    # Constructor: Initializes the SentenceTransformer model and sets a random seed for reproducibility
    def __init__(self, model=None):
        random.seed(10)  # Setting a fixed random seed to ensure consistent results
        self.word_dist_embedding = {}  # Dictionary to cache word embeddings
        # If no model is provided, use the default model name from the config
        if model is None:
            self.model = SentenceTransformer(MODEL_NAME)  # Load the pre-trained SentenceTransformer model
        else:
            self.model = SentenceTransformer(model)  # Load a specified model

    # Method to get the BERT embedding for a given word or sentence
    def get_bert_embed(self, word: str):
        # Check if the embedding is already cached
        embed = self.word_dist_embedding.get(word)
        if embed is not None:
            return embed  # Return the cached embedding
        # If not cached, create the embedding and store it
        embed = self.create_embedding(word)
        self.word_dist_embedding[word] = embed
        return embed

    # Method to create and normalize a new embedding using the SentenceTransformer model
    def create_embedding(self, word: str):
        # Generate and normalize the embedding for the input word or sentence
        sentence_embeddings = self.model.encode(word, show_progress_bar=False, normalize_embeddings=True)
        return sentence_embeddings
