# Interpreting Embedding Spaces by Conceptualization

The primary task here is to enhance the interpretability of embeddings produced
by large language models (LLMs), such as BERT and GPT-2, using a novel Conceptual Embedding
Space (CES) framework. The research question
we seek to answer is: Can we transform complex,
high dimensional latent embeddings into an interpretable conceptual space without compromising
their representational power? Focus is on developing a method that maps latent embeddings
to a set of human understandable concepts, facilitating a better understanding of model behavior and improving transparency in language model
decision making.

We have run, verified and thoroughly annotated the files: full_llm.py, feature_generation.py, model_embedding.py, classification_test.py, core_on_demand.py such that we clearly associate methodology used in the paper by the authors to the actual code.

### Steps to run this code:
1. create an environment with environment.yml in the additional files
2. `cd src`
3. `CUDA_VISIBLE_DEVICES="" python run_tests.py -p ../additional_files/ --human_and_model_evaluation --classification_test --triplets_test --example_creation --model_application --full_llm_explained`

To see help about the arguments run `python run_tests.py -h`


### How to cite this work:
This work has been accepted to the 2023 EMNLP conference. If you use this code, please cite our paper:
```@article{simhi2022interpreting,
  title={Interpreting Embedding Spaces by Conceptualization},
  author={Simhi, Adi and Markovitch, Shaul},
  journal={arXiv preprint arXiv:2209.00445},
  year={2022}
}
```
