import logging
from typing import List, Tuple
import networkx as nx  # Used for graph representation and traversal
from scipy.spatial import distance  # Used to compute distances between vectors
from scipy.stats import rankdata
from sklearn.datasets import fetch_20newsgroups  # Example dataset for testing
from sklearn.ensemble import RandomForestClassifier  # Example ML model
from sklearn.model_selection import train_test_split, StratifiedKFold  # Data splitting utilities

from model_embedding import BlackModelEmbed  # Custom module for embedding extraction
import numpy as np
from config import START_SEQ  # Custom configuration for starting sequences
from using_pickle import from_yaml_to_python, read_zip  # Helper functions for loading data
from sklearn import tree  # ML model for decision trees

# Main class that defines the core on demand functionality
class coreOnDemand():

    # Constructor: Initializes embeddings, loads the graph, and sets parameters
    def __init__(self, core_all_depth: List[List[str]], graph_g_path):
        self.all_cores = core_all_depth  # Core categories at different depths
        self.sbert = BlackModelEmbed()  # Instance of the embedding model
        self.sbert_core_embedding = {}  # Stores core embeddings without context
        self.sbert_core_embedding_with_context = {}  # Stores core embeddings with context
        self.g_graph = read_zip(graph_g_path)  # Loads the graph from a zip file
        self.percentage_of_children_to_open = 0.5  # Percentage of children categories to open

        # Generate embeddings for each core category
        for core in self.all_cores:
            for c in core:
                # Create embeddings for the core category and its context
                self.sbert_core_embedding[c] = self.sbert.get_bert_embed(c.replace("Category:", START_SEQ).replace("_", " "))
                self.sbert_core_embedding_with_context[c] = self.sbert.get_bert_embed(self.concept_name_with_its_children([c])[0])
        self.sbert_core_embedding_keys = self.sbert_core_embedding.keys()

    # Generates interpretable embeddings for a given sentence using a transformation matrix
    def get_features_explainable_sbert(self, matrix, sentence: str) -> np.ndarray:
        embedding = self.sbert.get_bert_embed(sentence)  # Get sentence embedding
        return matrix.dot(embedding)  # Transform embedding using the matrix

    # Constructs a concept name along with its children (contextualization)
    def concept_name_with_its_children(self, list_of_concepts: List[str]) -> List[str]:
        new_concept_names = []
        for c in list_of_concepts:
            children = self.choose_children_to_open(c, 1)  # Select children categories
            children = children[:min(len(children), 2)]  # Limit to two children
            if len(children) < 2:
                new_concept_names.append(c)
            else:
                # Create a context-enhanced concept name
                new_concept_names.append(c.replace("Category:", START_SEQ).replace("_", " ") +
                                         " such as " + children[0].replace("Category:", START_SEQ).replace("_", " ") +
                                         " and " + children[1].replace("Category:", START_SEQ).replace("_", " "))
        return new_concept_names

    # Main method to generate a "core on demand" set of categories
    def create_core_on_demand(self, X_train, number_of_classes=None, y_train=None, w=0.8, N=1000, min_dist=None, max_size=None, remove_p=False) -> List[str]:
        core_opend = []  # List of opened core categories
        core_names = self.all_cores[0][:]  # Start with the first set of core categories
        core_embedding = np.array([self.sbert_core_embedding.get(c) for c in core_names])
        core_on_train = [self.get_features_explainable_sbert(core_embedding, x) for x in X_train]

        # Calculate entropy if number_of_classes is provided (used for classification tasks)
        if number_of_classes is not None:
            entropy_core = self.calculate_entropy(core_names, y_train, core_on_train, number_of_classes)

        assert len(core_names) == len(core_embedding)  # Ensure embeddings are correctly matched

        for i in range(N):  # Iterate up to N times or until stopping conditions are met
            if max_size is not None and len(core_names) >= max_size:
                break

            average_train_vector = np.average(np.array(core_on_train), axis=0)  # Compute average vector
            if number_of_classes is not None:
                entropy_mul_score = entropy_core * (1 - w) + w * average_train_vector  # Weighted score
            else:
                entropy_mul_score = average_train_vector

            core_to_expand = self.find_core_to_expend(core_opend, entropy_mul_score, core_names)  # Find next core to expand
            if core_to_expand == "":
                break

            core_opend.append(core_to_expand)  # Add the core to the list of opened cores
            core_children = self.choose_children_to_open(core_to_expand, self.percentage_of_children_to_open)
            core_children = [c for c in core_children if c not in core_names and c in self.sbert_core_embedding_keys]

            core_names += core_children  # Add new children to core names
            if remove_p is True and len(core_children) > 0:  # Optionally remove the parent core
                core_names.remove(core_to_expand)
            
            # Update embeddings and train vectors
            core_embedding = np.array([self.sbert_core_embedding.get(c) for c in core_names])
            core_on_train = [self.get_features_explainable_sbert(core_embedding, x) for x in X_train]

            if number_of_classes is not None and len(core_children) > 0:
                new_entropy_core = self.calculate_entropy(core_names, y_train, core_on_train, number_of_classes)
                if np.average(new_entropy_core) <= 1 / number_of_classes and max_size is None:
                    break
                entropy_core = new_entropy_core

            if min_dist is not None and len(core_children) > 0:  # Check distance condition for stopping
                distence_min = sorted(set(distance.cdist(np.array(core_on_train), np.array(core_on_train), 'cosine').flatten()))[0]
                if distence_min > min_dist:
                    break

        print(f"final core length {len(core_names)}")
        return core_names

    # Calculates the entropy for each core category
    def calculate_entropy(self, core_names, y_train, core_on_train, number_of_classes):
        classes_per_core = [[] for _ in range(len(core_names))]
        for i, y in enumerate(y_train):
            core_max = core_on_train[i].argmax()  # Find the core category with the highest score
            classes_per_core[core_max].append(y)
        entropy_core = np.array([len(set(classes)) / number_of_classes for classes in classes_per_core])
        return entropy_core

    # Finds the best core category to expand based on the scores
    def find_core_to_expend(self, list_of_core_opened: List[str], score_core: np.ndarray, core_list: List[str]) -> str:
        score_core_arg = score_core.argsort()[::-1]  # Sort scores in descending order
        best_core = ""
        for score in score_core_arg:
            if core_list[score] not in list_of_core_opened:
                best_core = core_list[score]
                break
        return best_core

    # Selects children of a core category to expand
    def choose_children_to_open(self, node_name, number_to_choose):
        final_children = []
        node_edges = self.g_graph.out_edges(node_name, data=True)  # Get edges from the graph
        for node in node_edges:
            if "Category:" in node[1]:
                weight = node[2].get("weight", 1)  # Default weight is 1
                final_children.append((node[1], weight))
        final_children = sorted(final_children, key=lambda x: x[1])  # Sort by weight
        final_children = final_children[-round(number_to_choose * len(final_children)):]  # Select top children
        return [self.remove_non_ascii_chars(c[0]) for c in final_children]

    # Removes non-ASCII characters from a string
    def remove_non_ascii_chars(self, string):
        return ''.join([i if ord(i) < 128 else ' ' for i in string])
