import logging
import random

import numpy as np
from numpy.linalg import norm, matrix_rank

from model_embedding import BlackModelEmbed  # Custom embedding model for extracting BERT embeddings
from config import START_SEQ  # Configuration setting for handling category names

# Class for generating features based on BERT embeddings, used to interpret the CES framework
class FeatureGeneration():

    # Constructor: Initializes the feature generation process using core categories and BERT embeddings
    def __init__(self, core_categories_list, model=None):
        self.core_categories_list = core_categories_list  # List of core conceptual categories
        self.sbert = BlackModelEmbed(model)  # Instantiates the embedding model
        # Create an embedding matrix for the core categories, which is crucial for the CES transformation
        self.sbert_core_embedding_matrix = self.createEmbeddingToBertBaseUsingCore()

    # Method to create a matrix of embeddings for the core categories using BERT
    def createEmbeddingToBertBaseUsingCore(self):
        basisInExpEmbedding = []  # List to store embeddings for each core category
        for c in self.core_categories_list:
            # Generate embedding for each category, replacing prefixes and underscores as per the CES framework
            c_embedd = self.sbert.get_bert_embed(c.replace("Category:", START_SEQ).replace("_", " "))
            basisInExpEmbedding.append(c_embedd)  # Add the embedding to the basis list
        basisInExpEmbedding = np.array(basisInExpEmbedding)  # Convert to a NumPy array for matrix operations
        return basisInExpEmbedding

    # Method to extract standard BERT embeddings for a given sentence
    def get_features_Sbert(self, sentence: str) -> np.ndarray:
        embedding = self.sbert.get_bert_embed(sentence)  # Generate BERT embedding for the input sentence
        return embedding

    # Method to generate interpretable features by projecting the BERT embedding onto the core embedding matrix
    def get_features_explainable_sbert(self, sentence: str) -> np.ndarray:
        embedding = self.sbert.get_bert_embed(sentence)  # Generate BERT embedding for the input sentence
        # Project the embedding onto the core embedding space to create explainable features
        return self.sbert_core_embedding_matrix.dot(embedding)
