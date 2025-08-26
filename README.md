# Final Project

## Model 1: MLKNN

The notebooks mlknn_model.ipynb and nn_model.ipynb are the main notebooks that trained the model for the Java portion of the dataset. The nn_model.ipynb utilizes scikit-learn's implementation of TF-IDF to tokenize the comments, and then uses a neural network to do the classification. The mlknn_model.ipynb utilize scikit-learn's functions to train a multi label k nearest neighbors classifier on the data. 

## Model 2: CodeT5 Pre Trained Model

The directory codet5_model_code contains all of the code for the CodeT5 pretrained model. There is a local_kmeans_pytorch library embedded into the directory for convience. The codet5_model.py encapsulates the logic needed to fine-tune the codet5-base model for our use case, and use its pre-trained embeddings for useful input into a simple neural network. The codet5_training_notebook.ipynb is a simple notebook outlying how to use the model to train a classifier on the data.

## Build Instructions

### Step 1
Install [Anaconda](https://www.anaconda.com/download/success) if you don't have it already. Follow steps for your operating system.

### Step 2
Create a new enviroment using the provided environment.yml file
```
conda env create --f environment.yml
```

And you're done. All of the required dependencies will be installed.
