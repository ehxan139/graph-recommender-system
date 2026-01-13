"""
Model Training and Evaluation Utilities
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import time


class Trainer:
    """
    Trainer for graph-based recommendation models.

    Handles training loop, validation, early stopping, and checkpointing.
    """

    def __init__(
        self,
        model,
        data_loader,
        learning_rate: float = 0.001,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        early_stopping_patience: int = 10
    ):
        """
        Initialize trainer.

        Args:
            model: NGCF or LightGCN model
            data_loader: RecommenderDataLoader instance
            learning_rate: Learning rate for optimizer
            device: Device to train on
            early_stopping_patience: Epochs to wait before early stopping
        """
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.early_stopping_patience = early_stopping_patience

        # Create adjacency matrix
        self.adj_matrix = model.create_adjacency_matrix(data_loader.train_matrix)

        self.best_val_metric = 0
        self.patience_counter = 0
        self.train_history = {'loss': [], 'val_recall': [], 'val_ndcg': []}

    def train_epoch(self, batch_size: int = 1024) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            batch_size: Batch size for training

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0
        total_bpr_loss = 0
        total_reg_loss = 0
        n_batches = 0

        for users, pos_items, neg_items in self.data_loader.get_train_batches(batch_size):
            users_tensor = torch.LongTensor(users).to(self.device)
            pos_items_tensor = torch.LongTensor(pos_items).to(self.device)
            neg_items_tensor = torch.LongTensor(neg_items).to(self.device)

            # Forward pass
            if hasattr(self.model, 'graph_convolution'):
                # LightGCN
                (users_emb, pos_emb, neg_emb,
                 users_emb_ego, pos_emb_ego, neg_emb_ego) = self.model(
                    self.adj_matrix, users_tensor, pos_items_tensor, neg_items_tensor
                )
                loss, bpr_loss, reg_loss = self.model.compute_bpr_loss(
                    users_emb, pos_emb, neg_emb,
                    users_emb_ego, pos_emb_ego, neg_emb_ego
                )
            else:
                # NGCF
                users_emb, pos_emb, neg_emb = self.model(
                    self.adj_matrix, users_tensor, pos_items_tensor,
                    neg_items_tensor, drop_flag=True
                )
                loss, bpr_loss, reg_loss = self.model.compute_bpr_loss(
                    users_emb, pos_emb, neg_emb
                )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_bpr_loss += bpr_loss.item()
            total_reg_loss += reg_loss.item()
            n_batches += 1

        return {
            'loss': total_loss / n_batches,
            'bpr_loss': total_bpr_loss / n_batches,
            'reg_loss': total_reg_loss / n_batches
        }

    def evaluate(self, k: int = 20) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            k: Top-k for metrics

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()

        recalls, ndcgs, hits = [], [], []

        # Evaluate on test users
        for user_idx in self.data_loader.test_interactions.keys():
            # Get ground truth items
            test_items = set(self.data_loader.test_interactions[user_idx])
            train_items = set(self.data_loader.train_interactions.get(user_idx, []))

            if len(test_items) == 0:
                continue

            # Get recommendations
            user_tensor = torch.LongTensor([user_idx]).to(self.device)
            scores = self.model.predict(self.adj_matrix, user_tensor).squeeze()

            # Exclude training items
            scores[list(train_items)] = -np.inf

            # Get top-k recommendations
            _, top_k_items = torch.topk(scores, k)
            top_k_items = set(top_k_items.cpu().numpy())

            # Calculate metrics
            hits_at_k = len(top_k_items & test_items)
            recall = hits_at_k / len(test_items)
            hit_rate = 1.0 if hits_at_k > 0 else 0.0

            # NDCG
            dcg = 0
            idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(test_items), k))])
            top_k_list = list(top_k_items)
            for i, item in enumerate(top_k_list):
                if item in test_items:
                    dcg += 1.0 / np.log2(i + 2)
            ndcg = dcg / idcg if idcg > 0 else 0

            recalls.append(recall)
            ndcgs.append(ndcg)
            hits.append(hit_rate)

        return {
            'recall@{}'.format(k): np.mean(recalls),
            'ndcg@{}'.format(k): np.mean(ndcgs),
            'hit_rate@{}'.format(k): np.mean(hits)
        }

    def fit(
        self,
        epochs: int = 100,
        batch_size: int = 1024,
        eval_every: int = 5,
        k: int = 20,
        verbose: bool = True
    ) -> Dict:
        """
        Train model for multiple epochs.

        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            eval_every: Evaluate every N epochs
            k: Top-k for evaluation
            verbose: Print progress

        Returns:
            Training history
        """
        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(batch_size)
            self.train_history['loss'].append(train_metrics['loss'])

            # Evaluate
            if epoch % eval_every == 0:
                eval_metrics = self.evaluate(k)
                self.train_history['val_recall'].append(eval_metrics[f'recall@{k}'])
                self.train_history['val_ndcg'].append(eval_metrics[f'ndcg@{k}'])

                if verbose:
                    print(f"Epoch {epoch}/{epochs} ({time.time()-start_time:.2f}s)")
                    print(f"  Loss: {train_metrics['loss']:.4f}")
                    print(f"  Recall@{k}: {eval_metrics[f'recall@{k}']:.4f}")
                    print(f"  NDCG@{k}: {eval_metrics[f'ndcg@{k}']:.4f}")

                # Early stopping
                current_metric = eval_metrics[f'recall@{k}']
                if current_metric > self.best_val_metric:
                    self.best_val_metric = current_metric
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            else:
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}/{epochs} - Loss: {train_metrics['loss']:.4f}")

        return self.train_history


def compute_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 20
) -> Dict[str, float]:
    """
    Compute ranking metrics.

    Args:
        predictions: Array of predicted item IDs (sorted by score)
        ground_truth: Array of ground truth item IDs
        k: Top-k cutoff

    Returns:
        Dictionary of metrics
    """
    predictions = predictions[:k]
    ground_truth_set = set(ground_truth)

    # Recall@k
    hits = len(set(predictions) & ground_truth_set)
    recall = hits / len(ground_truth_set) if len(ground_truth_set) > 0 else 0

    # Precision@k
    precision = hits / k

    # Hit Rate
    hit_rate = 1.0 if hits > 0 else 0.0

    # NDCG@k
    dcg = 0
    for i, item in enumerate(predictions):
        if item in ground_truth_set:
            dcg += 1.0 / np.log2(i + 2)

    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k))])
    ndcg = dcg / idcg if idcg > 0 else 0

    return {
        'recall': recall,
        'precision': precision,
        'hit_rate': hit_rate,
        'ndcg': ndcg
    }
