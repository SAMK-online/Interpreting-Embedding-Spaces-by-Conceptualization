import csv
import logging
import random
from typing import List, Tuple
import numpy as np

from datasets import load_dataset  # Hugging Face datasets library for loading datasets
from sklearn.datasets import fetch_20newsgroups  # Sci-kit learn library for fetching the 20 Newsgroups dataset
from sklearn.ensemble import RandomForestClassifier  # Random forest classifier from Sci-kit learn

from core_on_demand import coreOnDemand  # Importing the coreOnDemand class for creating core categories
from sklearn.model_selection import StratifiedKFold  # For stratified k-fold cross-validation
from feature_generation import FeatureGeneration  # For feature generation using core concepts
from config import START_SEQ  # Configuration setting for starting sequences

# Class for evaluating the performance of the CES framework against a traditional ML model
class ClassificationCheckModelVsInterpretableModel():
    
    # Constructor: Initializes paths, core categories, and sets up coreOnDemand
    def __init__(self, classification_path, core_all_depth: List[List[str]], graph_g_path):
        self.all_cores = core_all_depth  # List of all core categories at different depths
        self.classification_path = classification_path  # Paths to classification datasets
        self.coreOnDemand = coreOnDemand(core_all_depth, graph_g_path)  # Instance of coreOnDemand
        self.n_split = 10  # Number of splits for cross-validation
        self.g = graph_g_path  # Path to the graph data

    # Method to load data from a CSV file and preprocess it
    def get_data(self, index) -> List[Tuple]:
        x_y = []  # List to store (text, label) tuples
        sentence_length = []  # List to store sentence lengths
        xis = []  # List to store unique sentences
        with open(self.classification_path[index], newline='') as f:
            data = csv.reader(f, skipinitialspace=True)
            for i, row in enumerate(data):
                tuple_value = (row[0], row[1])  # Create a tuple of (text, label)
                if tuple_value not in x_y and row[0] not in xis:
                    x_y.append(tuple_value)
                    xis.append(row[0])
                    sentence_length.append(len(tuple_value[0].split(" ")))  # Calculate sentence length
        return x_y  # Return the list of tuples

    # Method to train and test a RandomForest classifier using provided feature functions
    def test_using_random_forest(self, X_train, X_test, y_train, y_test, core_names, func):
        x = [func(x) for x in X_train]  # Generate features for training data
        clf = RandomForestClassifier(max_depth=5, random_state=42)  # Initialize the classifier
        clf.fit(x, y_train)  # Train the classifier
        return [round(clf.score([func(x) for x in X_test], y_test), 2)], list(clf.predict([func(x) for x in X_test]))

    # Method to fetch and preprocess the 20 Newsgroups dataset
    def get_data_20news(self):
        all_data = fetch_20newsgroups(data_home="../additional_files/", subset="all")
        xis = all_data.data  # Extract text data
        xis = [x.split("\nSubject:")[-1] for x in xis]  # Remove subject headers
        yis = all_data.target  # Extract labels
        y_classes = all_data.target_names  # Get class names
        y_real = [y_classes[y] for y in yis]  # Convert labels to class names
        c = list(zip(xis, yis))  # Combine text and labels into tuples
        self.random_seed()  # Set random seed for reproducibility
        random.shuffle(c)  # Shuffle the data
        X_train, y_train = zip(*c)
        return X_train[:min(len(X_train), 10000)], y_train[:min(len(X_train), 10000)]  # Return a subset

    # Method to load and preprocess the DBPedia dataset
    def get_dbpedia_data(self):
        train = load_dataset("dbpedia_14")["train"]
        X_train = [word.get("title") + " " + word.get('content') for word in train]  # Concatenate title and content
        y_train = [word.get('label') for word in train]  # Extract labels
        c = list(zip(X_train, y_train))
        self.random_seed()
        random.shuffle(c)
        X_train, y_train = zip(*c)
        return X_train[:min(len(X_train), 10000)], y_train[:min(len(X_train), 10000)]  # Return a subset

    # Method to load and preprocess the AG News dataset
    def get_ag_news_data(self):
        train = load_dataset("ag_news")["train"]
        X_train = [word.get("text") for word in train]  # Extract text data
        y_train = [word.get('label') for word in train]  # Extract labels
        c = list(zip(X_train, y_train))
        self.random_seed()
        random.shuffle(c)
        X_train, y_train = zip(*c)
        return X_train[:min(len(X_train), 10000)], y_train[:min(len(X_train), 10000)]  # Return a subset

    # Method to load custom datasets from CSV files
    def get_data_else(self, index):
        data_processed = self.get_data(index)
        self.random_seed()
        all_x_all_y = random.sample(data_processed, min(len(data_processed), 10000))  # Randomly sample the data
        all_x = [x_y[0] for x_y in all_x_all_y]  # Extract text
        all_y = [x_y[1] for x_y in all_x_all_y]  # Extract labels
        classes = list(set(all_y))  # Get unique classes
        all_y = [classes.index(y) for y in all_y]  # Convert labels to numeric indices
        return all_x, all_y

    # Method to set a fixed random seed for reproducibility
    def random_seed(self):
        random.seed(42)

    # Method to separate data into training and testing sets based on provided indices
    def get_data_from_index(self, all_x: List, all_y: List, indexes_list: List[int]) -> Tuple[List, List]:
        x = [all_x[index] for index in indexes_list]  # Get text data
        y = [all_y[index] for index in indexes_list]  # Get label data
        return x, y

    # Method to calculate Cohen's Kappa and accuracy
    def kapper_test(self, y1, y2):
        p0 = sum([1 for j in range(len(y1)) if y1[j] == y2[j]]) / len(y1)  # Proportion of agreement
        classes = list(set(y1).union(y2))  # Unique classes from both sets
        pe = 0
        for k in classes:
            pe += sum([1 for j in range(len(y1)) if y1[j] == k]) * sum([1 for j in range(len(y2)) if y2[j] == k])
        pe = pe / len(y1) ** 2  # Expected agreement
        k = (p0 - pe) / (1 - pe)  # Cohen's Kappa calculation
        return p0, k

    # Method to calculate random agreement for comparison
    def random_kapper_test(self, y1):
        classes = list(set(y1))
        y2 = [random.choice(classes) for _ in y1]  # Generate random predictions
        p0 = sum([1 for j in range(len(y1)) if y1[j] == y2[j]]) / len(y1)  # Proportion of agreement
        return p0

    # Method to evaluate the CES framework and a standard model using cross-validation
    def check_accuracy_all_models(self):
        skf = StratifiedKFold(n_splits=self.n_split, random_state=42, shuffle=True)
        icos_acc = []  # List to store accuracies for the CES framework
        model_acc = []  # List to store accuracies for the standard model
        p0_andk = []  # List to store Cohen's Kappa results
        raw_random = []  # List to store random agreement results
        for train_index, test_index in skf.split(self.all_x, self.all_y):
            x_t, y_t = self.get_data_from_index(self.all_x, self.all_y, train_index)
            x_test, y_test = self.get_data_from_index(self.all_x, self.all_y, test_index)
            
            # Create core categories on demand
            self.core_on_demand = self.coreOnDemand.create_core_on_demand(x_t, len(set(self.all_y)), y_t, N=1000, max_size=768)
            self.core_on_demand = [c.replace("Category:", START_SEQ).replace("_", " ") for c in self.core_on_demand]
            self.feature_generation = FeatureGeneration(self.core_on_demand)  # Generate features

            # Evaluate using RandomForest with explainable features
            icosa, predict_icos = self.test_using_random_forest(
                x_t, x_test, y_t, y_test, self.core_on_demand, self.feature_generation.get_features_explainable_sbert
            )
            # Evaluate using RandomForest with standard BERT features
            modela, predict_mode = self.test_using_random_forest(
                x_t, x_test, y_t, y_test, self.core_on_demand, self.feature_generation.get_features_Sbert
            )
            icos_acc += icosa
            model_acc += modela
            p0_andk.append(self.kapper_test(predict_icos, predict_mode))  # Calculate agreement
            raw_random.append(self.random_kapper_test(predict_mode))  # Calculate random agreement

        # Print average accuracies and agreement scores
        print(
            f"p0 is {np.average([i[0] for i in p0_andk])} with s.d {np.std([i[0] for i in p0_andk])}\n"
            f"Coefficient {np.average([i[1] for i in p0_andk])} with s.d {np.std([i[1] for i in p0_andk])}\n"
            f"random p0 is {np.average(raw_random)} with s.d {np.std(raw_random)}\n"
            f"icos accuracy is {np.average(icos_acc)} with s.d {np.std(icos_acc)}\n"
            f"model accuracy is {np.average(model_acc)} with s.d {np.std(model_acc)}\n"
        )

    # Method to run classification checks on standard datasets
    def check_accuracy_on_librery_dataset(self):
        # Run classification on the 20 Newsgroups dataset
        self.all_x, self.all_y = self.get_data_20news()
        self.check_accuracy_all_models()
        # Run classification on the AG News dataset
        self.all_x, self.all_y = self.get_ag_news_data()
        self.check_accuracy_all_models()
        # Run classification on the DBPedia dataset
        self.all_x, self.all_y = self.get_dbpedia_data()
        self.check_accuracy_all_models()

    # Method to run classification checks on all datasets, including custom ones
    def run_on_all_datasets(self):
        self.check_accuracy_on_librery_dataset()  # Check accuracy on standard datasets
        for index, data in enumerate(self.classification_path):
            self.all_x, self.all_y = self.get_data_else(index)  # Load custom dataset
            data_together = list(zip(self.all_x, self.all_y))
            self.random_seed()
            random.shuffle(data_together)
            self.all_x, self.all_y = zip(*data_together)
            self.check_accuracy_all_models()  # Check accuracy on custom dataset
