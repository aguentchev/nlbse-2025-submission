import pandas as pd
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, RobertaTokenizer
from tqdm.auto import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

from local_kmeans_pytorch import kmeans, kmeans_predict

import wandb


class LinearNN(torch.nn.Module):
    """
    A simple Pytorch multilayer perceptron intended to be the final step in the codet5-base model.

    Attributes
    ----------
    layers: torch.nn.Sequential
        Defines the layers of the simple neural network

    Methods
    ----------
    forward: torch.Tensor
        Runs a forward pass on the model. This method is called when you use the class as a function (LinearNN(X))
    """
    def __init__(self, input_dims, output_dims) -> None:
        super(LinearNN, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dims * 2, 2 ** 10),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(2 ** 10, 2 ** 9),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(2 ** 9, output_dims)
        )
    
    def forward(self, x):
        return self.layers(x)

class CodeClassificationModel(torch.nn.Module):
    def __init__(self, pre_trained_model_name, lang, output_dims, max_sequence_length = 128, batch_size = 16, learning_rate = 5e-5, epochs = 10, device="cuda", num_cluster = 8) -> None:
        """
        Initializes the CodeClassificationModel with configuration parameters.

        Parameters
        ----------
        pre_trained_model_name : str
            Name of the pre-trained Code T5 model from hugging face.
        lang : str
            Programming language for the model to train on.
        output_dims : int
            Number of output dimensions for classification.
        max_sequence_length : int, optional
            Maximum token sequence length
        batch_size : int, optional
            Number of samples per batch.
        learning_rate : float, optional
            Learning rate for the optimizer.
        epochs : int, optional
            Number of training epochs.
        device : str, optional
            Device to run computations on.
        num_cluster : int, optional
            Number of clusters for KMeans.
        """
        super(CodeClassificationModel, self).__init__()
        self.config = {
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "language": lang,
            "hidden_layers": [2 ** 10, 2 ** 9],
            "pretrained_base": pre_trained_model_name
        }
        self.run = wandb.init(
            project=f"NLBSE2025-CodeCommentClassification",
            config=self.config
        )
        self.max_sequence_length = max_sequence_length
        self.model_name = pre_trained_model_name
        self.batch_size = batch_size
        self.model = T5ForConditionalGeneration.from_pretrained(pre_trained_model_name).to(device)
        self._freeze_first_n_layers(len(self.model.encoder.block) - 3)
        self.tokenizer = RobertaTokenizer.from_pretrained(pre_trained_model_name)
        self.lang = lang
        hidden_size = self.model.config.hidden_size
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.fc_model = LinearNN(hidden_size, output_dims).to(device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        self.num_clusters = num_cluster

        self.labels = {
            'java': ['summary', 'Ownership', 'Expand', 'usage', 'Pointer', 'deprecation', 'rational'],
            'python': ['Usage', 'Parameters', 'DevelopmentNotes', 'Expand', 'Summary'],
            'pharo': ['Keyimplementationpoints', 'Example', 'Responsibilities', 'Classreferences', 'Intent', 'Keymessages', 'Collaborators']
        }
        self.dataset = load_dataset('NLBSE/nlbse25-code-comment-classification')
        self.tokenized_datasets = {}


    def _freeze_first_n_layers(self, n):
        """
        Freezes the first n encoder layers of the T5 model.

        Parameters
        ----------
        n : int
            Number of encoder layers to freeze.
        """
        for param in self.model.encoder.block[:n].parameters():
            param.requires_grad = False

    def transformer_forward(self, inputs, attention_mask):
        """
        Runs the forward pass through the encoder and applies mean pooling.

        Parameters
        ----------
        inputs : torch.Tensor
            Tokenized input IDs for the model.
        attention_mask : torch.Tensor
            Attention mask indicating valid tokens.

        Returns
        -------
        torch.Tensor
            Mean pooled encoder output.
        """
        embeddings = self.model.encoder(input_ids=inputs, attention_mask=attention_mask).last_hidden_state
        return embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)  # Mean pooling

    def forward(self, inputs, attention_mask, device="cuda"):
        """
        Completes the full forward pass through the model, including clustering and classification.

        Parameters
        ----------
        inputs : torch.Tensor
            Tokenized input IDs for the model.
        attention_mask : torch.Tensor
            Attention mask indicating valid tokens.
        device : str, optional
            Device to run computations on.

        Returns
        -------
        torch.Tensor
            Final classification logits.
        """
        pooled = self.transformer_forward(inputs=inputs, attention_mask=attention_mask)
        print(pooled.shape)

        if pooled.dim() == 1:
            pooled = torch.unsqueeze(pooled, 0)
        
        cluster_ids_x = kmeans_predict(pooled, self.cluster_centers, device=device, tqdm_flag=False)
        cluster_features = self.cluster_centers[cluster_ids_x].to(device)
        combined_representations = torch.cat((pooled, cluster_features), dim=1)

        return self.fc_model(combined_representations)
    
    def get_data_loader(self, lang, split):
        """
        Prepares a DataLoader for the specified dataset split. Uses the tokenized datasets from preprocess_dataset for its data

        Parameters
        ----------
        lang : str
            Target programming language.
        split : str
            Dataset split to load ('train', 'validation', or 'test').

        Returns
        -------
        DataLoader
            DataLoader object for the specified dataset split.
        """
        return DataLoader(
            self.tokenized_datasets[f"{lang}_{split}"],
            batch_size=self.batch_size,
            shuffle=split == "train"
        )
    
    def preprocess_dataset(self, val_percentage=0.1):
        """
        Tokenizes and formats the dataset for training and evaluation.

        Parameters
        ----------
        val_percentage : float, optional
            Percentage of training data used for validation.
        """
        def preprocess_example(examples):
            tokens = self.tokenizer(
                examples["combo"],
                truncation=True,
                padding="max_length",
                max_length=self.max_sequence_length
            )
            tokens["labels"] = examples["labels"]
            return tokens
        
        self.tokenized_datasets[f"{self.lang}_train"] = self.dataset[f"{self.lang}_train"].map(
                preprocess_example, batched=True)
        self.tokenized_datasets[f"{self.lang}_test"] = self.dataset[f"{self.lang}_test"].map(
            preprocess_example, batched=True)
        
        self.tokenized_datasets[f"{self.lang}_train"].set_format(
            type="torch", columns=['input_ids', 'attention_mask', 'labels'])

        self.tokenized_datasets[f"{self.lang}_test"].set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"])

        train_valid_split = self.tokenized_datasets[f"{self.lang}_train"].train_test_split(test_size=val_percentage)
        self.tokenized_datasets[f"{self.lang}_train"] = train_valid_split["train"]
        self.tokenized_datasets[f"{self.lang}_validation"] = train_valid_split["test"]

    def get_preds(self, output):
        """
        Converts raw model logits into binary classification predictions.

        Parameters
        ----------
        output : torch.Tensor
            Model output logits.

        Returns
        -------
        torch.Tensor
            Binary classification predictions.
        """
        probs = torch.sigmoid(output)
        return (probs > 0.5).int().cpu()
    
    def save_model(self, output_dir=None, epoch=None, save_wandb_artifact=False):
        """
        Saves the model's state and clustering centers to a file.

        Parameters
        ----------
        output_dir : str, optional
            Directory where the model will be saved.
        epoch : int, optional
            Current training epoch.
        save_wandb_artifact : bool, optional
            Whether to save the model as a Weights & Biases artifact.
        """
        model_file_name = f"model_path{f"_{epoch + 1}" if epoch is not None else ""}_{self.lang}.pth"
        model_output_path = model_file_name if output_dir is None else f"{output_dir}/{model_file_name}"
        complete_state_dict = {
            "state_dict": self.state_dict(),
            "cluster_centers": self.cluster_centers
        }
        torch.save(complete_state_dict, model_output_path)
        if save_wandb_artifact:
            model_artifact = wandb.Artifact(
                f"code-comment-classification-t5-fine-tune-{self.lang}",
                "model",
                f"Classification model for {self.lang}"
            )
            model_artifact.add_file(model_output_path)
            wandb.log_artifact(model_artifact)

    def load_model(self, complete_state_dict: dict):
        """
        Loads the model's state and clustering centers from a saved file.

        Parameters
        ----------
        complete_state_dict : dict
            Dictionary containing model state and cluster centers.
        """
        self.load_state_dict(complete_state_dict["state_dict"])
        self.cluster_centers = complete_state_dict["cluster_centers"]
    
    def train_model(self, debug=False, device="cuda"):
        """
        The main training loop. Trains the model.

        Parameters
        ----------
        debug : bool, optional
            Whether to print debugging information.
        device : str, optional
            Device to run computations on.
        """
        # Ensure model is in train mode
        epochs = self.config["epochs"]
        self.train()
        self.to(device)
        train_loader = self.get_data_loader(self.lang, "train")
        validation_loader = self.get_data_loader(self.lang, "validation")

        for epoch in tqdm(range(epochs)):
            """
            |-------------------------|
             Main Training Loop
            |-------------------------|
            """
            total_train_loss = 0.0
            # Gather all training samples and predictions made on those samples to calculate metrics
            y_train_trues = []
            y_train_preds = []
            for batch in tqdm(train_loader):
                self.optimizer.zero_grad()

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].float().to(device)

                _, self.cluster_centers = kmeans(self.transformer_forward(input_ids, attention_mask), self.num_clusters, tqdm_flag=False, device=device)
                outputs = self(inputs=input_ids, attention_mask=attention_mask)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item()

                preds = self.get_preds(outputs)
                y_train_preds.append(preds)
                y_train_trues.append(labels.cpu())
            
            # Concatanate everything and calculate the metrics for wandb dashboard
            avg_train_loss = total_train_loss / len(train_loader)
            y_train_pred = torch.concat(y_train_preds).numpy()
            y_train_true = torch.concat(y_train_trues).numpy()
            train_metrics = self.get_metrics(y_train_pred, y_train_true, "train", avg_train_loss)

            """
            |-------------------------|
             Validation Metric Section
            |-------------------------|
            """
            total_val_loss = 0.0
            self.eval()
            y_val_preds = []
            y_val_trues = []
            with torch.no_grad():
                for batch in tqdm(validation_loader, "Validation Eval"):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].float().to(device)

                    outputs = self(input_ids, attention_mask=attention_mask)
                    loss = self.loss_fn(outputs, labels)

                    total_val_loss += loss.item()

                    preds = self.get_preds(outputs)
                    y_val_preds.append(preds)
                    y_val_trues.append(labels.cpu())

            """
            |-------------------------|
             Weights and Biases Logging
            |-------------------------|
            """

            avg_val_loss = total_val_loss / len(train_loader)
            y_val_pred = torch.concat(y_train_preds).numpy()
            y_val_true = torch.concat(y_train_trues).numpy()
            val_metrics = self.get_metrics(y_val_pred, y_val_true, "validation", avg_val_loss)
            all_metrics = {**train_metrics, **val_metrics}
            wandb.log(all_metrics, step=epoch + 1)
            self.save_model(None, None, True)
            if debug:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss}")
            # Set back to train mode for next epoch
            self.train()

    def get_metrics(self, y_pred, y_test, split, loss):
        """
        Computes all of the evaluation metrics for the competition.

        Parameters
        ----------
        y_pred : np.ndarray
            Model predictions.
        y_test : np.ndarray
            Ground truth labels.
        split : str
            Dataset split being evaluated ('train', 'validation', or 'test').
        loss : float
            Computed loss for the dataset split.

        Returns
        -------
        dict
            Dictionary of computed evaluation metrics.
        """
        metrics = {}
        for i, category in enumerate(self.labels[self.lang]):
            assert(len(y_pred[i]) == len(y_test[i]))
            tp = np.sum((y_test[:, i] == 1) & (y_pred[:, i] == 1), dtype=np.float64)
            tn = np.sum((y_test[:, i] == 0) & (y_pred[:, i] == 0), dtype=np.float64)
            fp = np.sum((y_test[:, i] == 0) & (y_pred[:, i] == 1), dtype=np.float64)
            fn = np.sum((y_test[:, i] == 1) & (y_pred[:, i] == 0), dtype=np.float64)
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = (2*tp) / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

            metrics[f"{self.lang}_{category}_{split}_precision"] = precision
            metrics[f"{self.lang}_{category}__{split}_recall"] = recall
            metrics[f"{self.lang}_{category}_{split}_f1"] = f1
            metrics[f"{self.lang}_{category}_{split}_accuracy"] = accuracy
        metrics[f"{self.lang}_{split}_loss"] = loss
        return metrics
    
    def get_metrics_df(self, y_pred, y_test):
        """
        Returns evaluation metrics as a DataFrame for easy plotting and table creation.

        Parameters
        ----------
        y_pred : np.ndarray
            Model predictions.
        y_test : np.ndarray
            Ground truth labels.

        Returns
        -------
        pd.DataFrame
            DataFrame containing evaluation metrics.
        """
        scores = []
        for i, category in enumerate(self.labels[self.lang]):
            assert(len(y_pred[i]) == len(y_test[i]))
            tp = np.sum((y_test[:, i] == 1) & (y_pred[:, i] == 1), dtype=np.float64)
            tn = np.sum((y_test[:, i] == 0) & (y_pred[:, i] == 0), dtype=np.float64)
            fp = np.sum((y_test[:, i] == 0) & (y_pred[:, i] == 1), dtype=np.float64)
            fn = np.sum((y_test[:, i] == 1) & (y_pred[:, i] == 0), dtype=np.float64)
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = (2*tp) / (2*tp + fp + fn)
            scores.append({'lan': self.lang, 'cat': category, 'precision': precision,'recall': recall,'f1': f1, 'accuracy': accuracy})
        return pd.DataFrame(scores)
    
    def evaluate(self, dataset="test", device="cuda"):
        """
        Evaluates the model on a specified dataset split and prints metrics.

        Parameters
        ----------
        dataset : str, optional
            Dataset split to evaluate ('train', 'validation', or 'test').
        device : str, optional
            Device to run computations on.
        """
        # Put model in evaluation mode
        self.eval()
        test_loader = self.get_data_loader(self.lang, dataset)
        total_loss = 0.0

        with torch.no_grad():
            y_pred = []
            y_test = []
            for batch in tqdm(test_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].float().to(device)

                outputs = self(input_ids, attention_mask=attention_mask)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).int()

                y_pred.append(preds)
                y_test.append(labels.cpu())

        y_pred = torch.cat(y_pred).cpu().numpy()
        y_test = torch.cat(y_test).cpu().numpy()
        avg_loss = total_loss / len(test_loader)
        print(f"Test Loss: {avg_loss:.4f}")
        self.metrics_df = self.get_metrics_df(y_pred, y_test)


def recreate_plots_from_epochs(epochs: int, lang: str, output_dims: int):
    for i in tqdm(range(1, epochs + 1)):
        run = wandb.init()
        artifact = run.use_artifact(f'derpypenguin/NLBSE2025-CodeCommentClassification/code-comment-classification-t5-fine-tune-{lang}:v{i}', type='model')
        artifact_dir = artifact.download()
        model = CodeClassificationModel(
            "Salesforce/codet5-base",
            lang=lang,
            output_dims=output_dims,
            epochs=16
        )
        model.preprocess_dataset()
        model.load_state_dict(torch.load(f"{artifact_dir}/model_path_{lang}.pth", weights_only=True))
        model.evaluate(dataset="validation")
        model.metrics_df.to_csv(f"results/result_for_v{i}.csv")


def get_test_results(wandb_user: str, wandb_project: str, lang: str, version_number: int, output_dims: int, output_dir: str):
    # Run that same function but on the final test dataset
    run = wandb.init()
    artifact = run.use_artifact(f'{wandb_user}/{wandb_project}-{lang}:v{version_number}', type='model')
    artifact_dir = artifact.download()
    model = CodeClassificationModel(
        "Salesforce/codet5-base",
        lang=lang,
        output_dims=output_dims,
        epochs=16
    )
    model.preprocess_dataset()
    model.load_state_dict(torch.load(f"{artifact_dir}/model_path_{lang}.pth", weights_only=True))
    model.evaluate(dataset="test")
    model.metrics_df.to_csv(f"{output_dir}/test_dataset_result.csv")

def recreate_plots_from_checkpoints(results_dir: str, lang: str = "java", output_dir: str = "plots"):
    path = Path(results_dir)
    file_count = sum(1 for _ in path.glob('*'))  # Files and folders, not recursive
    epoch_list = list(range(1, file_count + 1))
    all_precision_data = {}
    all_recall_data = {}
    all_f1_data = {}
    all_accuracy_data = {}

    for i in epoch_list:
        file_name = path / f"result_for_v{i}.csv"
        df = pd.read_csv(file_name)
        for category, group in df.groupby("cat"):
            if category not in all_precision_data:
                all_precision_data[category] = []
                all_recall_data[category] = []
                all_f1_data[category] = []
                all_accuracy_data[category] = []
            all_precision_data[category].append(group["precision"].iloc[0])
            all_recall_data[category].append(group["recall"].iloc[0])
            all_f1_data[category].append(group["f1"].iloc[0])
            all_accuracy_data[category].append(group["accuracy"].iloc[0])
    
    for category in all_precision_data.keys():
        plt.figure(figsize=(12, 8))
        plt.plot(epoch_list, all_precision_data[category], label="Precision")
        plt.plot(epoch_list, all_recall_data[category], label="Recall")
        plt.plot(epoch_list, all_f1_data[category], label="f1")
        plt.plot(epoch_list, all_accuracy_data[category], label="Accuracy")

        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(f"{str.capitalize(lang)} Metrics for {category}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/{category}_metrics.png")
        plt.close()