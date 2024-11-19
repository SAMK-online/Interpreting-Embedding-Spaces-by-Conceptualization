from typing import Tuple, List
import matplotlib.pyplot as plt  # Used for plotting graphs of layer-wise ranks
from numpy.linalg import norm  # Function to compute vector norms
from transformers import BertTokenizer, BertModel, BertConfig, GPT2Model, GPT2Tokenizer, GPT2Config  # Transformer models and configurations
import numpy as np

from core_on_demand import coreOnDemand  # Import coreOnDemand for creating core concepts

# Class for working with large language models (LLMs) to generate explainable embeddings
class FullLLM():

    # Constructor: Initializes the LLM, tokenizer, and creates embeddings for core concepts
    def __init__(self, g_graph, core_categories, model_name="bert-base-uncased", text=None):
        self.model_name = model_name  # LLM model name
        # If no specific text is provided, use default core categories
        if text is None:
            self.core = [c.replace("Category:", "") for c in core_categories[2]]
            print(len(self.core))  # Print the number of core categories
        else:
            # Create core categories on demand if text is provided
            core_on_demand = coreOnDemand(core_categories, g_graph)
            with open(text) as f:
                self.text = f.read()
            print(self.text)
            # Generate a custom set of core concepts from the text
            self.core = core_on_demand.create_core_on_demand(self.text.split(". ")[:10], max_size=768)
            self.core = [c.replace("Category:", "").replace("_", " ") for c in self.core if
                         'Wikipedia categories named after' not in c]

        # Load the specified LLM (BERT or GPT-2) and its tokenizer
        if self.model_name == "bert-base-uncased":
            config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name, config=config)
        else:
            # Load GPT-2 if specified
            config = GPT2Config.from_pretrained(self.model_name, output_hidden_states=True)
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.model = GPT2Model.from_pretrained(self.model_name, config=config)

        # Create embeddings for the core concepts
        self.embedding_core = self.create_embedding_for_core()

    # Method to create embeddings for the core concepts using the LLM
    def create_embedding_for_core(self) -> dict:
        embedding_core = {}
        for c in self.core:
            inputs = self.tokenizer(c, return_tensors="pt")  # Tokenize the core concept
            outputs = self.model(**inputs)  # Get model outputs, including hidden states
            embedding_core[c] = outputs[2]  # Store the hidden states as embeddings
        return embedding_core

    # Method to generate explainable embeddings and scores for a given sentence
    def get_explainable_embedding_with_score(self, sentence: str):
        inputs = self.tokenizer(sentence, return_tensors="pt")  # Tokenize the input sentence
        outputs = self.model(**inputs)  # Get model outputs, including hidden states
        embedding = outputs[2]  # Extract hidden states (layer-wise embeddings)
        print(f"{sentence:} {np.shape(outputs[0])} {len(embedding)} {np.shape(embedding[0])}")

        all_layers_embedd = []  # List to store similarity scores across layers

        # Compute similarity scores for each layer
        for i in range(len(embedding)):
            embedd = []
            # Average the embeddings of all tokens (excluding special tokens for BERT)
            if self.model_name == "bert-base-uncased":
                embedding_s_layer = embedding[i][0][1:-1].mean(0).detach().numpy()
            else:
                embedding_s_layer = embedding[i][0].mean(0).detach().numpy()

            # Compute cosine similarity between the sentence embedding and each core concept embedding
            for c, c_embedd in self.embedding_core.items():
                assert len(c_embedd) == len(embedding)  # Ensure the embeddings are aligned
                if self.model_name == "bert-base-uncased":
                    c_embedd_layer = c_embedd[i][0][1:-1].mean(0).detach().numpy()
                else:
                    c_embedd_layer = c_embedd[i][0].mean(0).detach().numpy()
                cosin_sim = np.dot(embedding_s_layer, c_embedd_layer) / (norm(c_embedd_layer) * norm(embedding_s_layer))
                embedd.append(cosin_sim)
            all_layers_embedd.append(embedd)

        # Get the top 3 concepts from the last and first layers
        top_3_concepts_index = np.array(all_layers_embedd[-1]).argsort()[-3:][::-1]
        bottom_3_concepts_index = np.array(all_layers_embedd[0]).argsort()[-3:][::-1]
        top_3_concepts_index = np.concatenate((top_3_concepts_index, bottom_3_concepts_index))
        top_3_concepts = [self.core[index] for index in top_3_concepts_index]
        print(f"{top_3_concepts=}")

        # Calculate rank of each top concept across layers
        top_3_concepts_rank = []
        for index in top_3_concepts_index:
            top_3_concepts_rank.append(
                [sum([1 for j in all_layers_embedd[i] if j > all_layers_embedd[i][index]]) for i in
                 range(len(all_layers_embedd))])
        print(f"{top_3_concepts_rank=} {np.shape(top_3_concepts_rank)}")

        return top_3_concepts_rank, top_3_concepts, all_layers_embedd

    # Method to generate and save a graph of concept ranks across layers for a given word
    def get_llm_graph_for_input(self, word: str):
        top_concept_rank, concept_names, all_layers_embedd = self.get_explainable_embedding_with_score(word)
        x = [i for i in range(1, 14)]  # List of layer indices (1 to 13 for BERT)
        
        # Plot the rank of each concept across layers
        for index, concept in enumerate(concept_names):
            plt.plot(x, [i + 1 for i in top_concept_rank[index]], label=concept)  # Plot ranks
            print(f"{concept=},{top_concept_rank[index]=}")
        
        plt.legend()  # Show legend with concept names
        plt.yscale("log")  # Use logarithmic scale for the y-axis
        plt.gca().invert_yaxis()  # Invert y-axis so that lower ranks appear higher
        plt.xlabel("Layers")  # Label for x-axis
        plt.ylabel("Rank")  # Label for y-axis
        plt.savefig(word + f"_llm_layers_model{self.model_name}.pdf", format="pdf")  # Save plot as PDF
        plt.close()
