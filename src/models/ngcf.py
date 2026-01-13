"""
Neural Graph Collaborative Filtering (NGCF) Model

Production-ready implementation of NGCF for recommendation systems.
This model leverages graph neural networks to capture high-order collaborative
signals in user-item interaction graphs.

Key Features:
- Graph convolution layers for message propagation
- Bi-interaction layers to capture feature interactions
- BPR loss optimization for ranking
- Efficient sparse matrix operations
- GPU acceleration support

Reference:
Wang et al. "Neural Graph Collaborative Filtering" (SIGIR 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict


class NGCF(nn.Module):
    """
    Neural Graph Collaborative Filtering model for collaborative filtering.

    The model uses graph neural networks to propagate embeddings through
    the user-item bipartite graph, capturing complex collaborative signals.

    Architecture:
    1. Embedding Layer: Initialize user and item embeddings
    2. Graph Convolution: Propagate embeddings through graph structure
    3. Bi-Interaction: Capture feature interactions between connected nodes
    4. Aggregation: Combine multi-hop embeddings for final representations
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_size: int = 64,
        layer_sizes: list = [64, 64, 64],
        node_dropout: float = 0.1,
        message_dropout: list = [0.1, 0.1, 0.1],
        regularization: float = 1e-5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize NGCF model.

        Args:
            n_users: Number of users in the dataset
            n_items: Number of items in the dataset
            embedding_size: Dimension of user/item embeddings
            layer_sizes: List of hidden layer dimensions for each GCN layer
            node_dropout: Dropout rate for graph adjacency matrix
            message_dropout: Dropout rates for each message passing layer
            regularization: L2 regularization coefficient
            device: Device to run model on ('cuda' or 'cpu')
        """
        super(NGCF, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_size = embedding_size
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.node_dropout = node_dropout
        self.message_dropout = message_dropout
        self.reg = regularization
        self.device = device

        # Initialize embeddings and weights
        self.embedding_dict, self.weight_dict = self._init_weights()

    def _init_weights(self) -> Tuple[nn.ParameterDict, nn.ParameterDict]:
        """
        Initialize model parameters using Xavier uniform initialization.

        Returns:
            Tuple of (embedding_dict, weight_dict) containing all trainable parameters
        """
        initializer = nn.init.xavier_uniform_

        # User and item embeddings
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_users, self.embedding_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_items, self.embedding_size)))
        })

        # Weights for graph convolution and bi-interaction layers
        weight_dict = nn.ParameterDict()
        layers = [self.embedding_size] + self.layer_sizes

        for k in range(self.n_layers):
            # Graph convolution weights: W1 in paper
            weight_dict.update({
                f'W_gc_{k}': nn.Parameter(initializer(torch.empty(layers[k], layers[k+1]))),
                f'b_gc_{k}': nn.Parameter(initializer(torch.empty(1, layers[k+1])))
            })

            # Bi-interaction weights: W2 in paper
            weight_dict.update({
                f'W_bi_{k}': nn.Parameter(initializer(torch.empty(layers[k], layers[k+1]))),
                f'b_bi_{k}': nn.Parameter(initializer(torch.empty(1, layers[k+1])))
            })

        return embedding_dict, weight_dict

    def create_adjacency_matrix(self, user_item_matrix: np.ndarray) -> torch.sparse.FloatTensor:
        """
        Create normalized adjacency matrix for the user-item bipartite graph.

        Uses symmetric normalization: D^(-1/2) * A * D^(-1/2)
        where A is the adjacency matrix and D is the degree matrix.

        Args:
            user_item_matrix: User-item interaction matrix (n_users x n_items)

        Returns:
            Normalized sparse adjacency tensor
        """
        import scipy.sparse as sp

        # Create adjacency matrix for bipartite graph
        # Structure: [[0, R], [R^T, 0]] where R is user-item interaction matrix
        adj_mat = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items),
            dtype=np.float32
        )
        adj_mat = adj_mat.tolil()

        R = user_item_matrix.tocoo()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        # Symmetric normalization
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
        norm_adj = norm_adj.tocoo()

        # Convert to PyTorch sparse tensor
        indices = torch.LongTensor([norm_adj.row, norm_adj.col])
        values = torch.FloatTensor(norm_adj.data)
        shape = torch.Size(norm_adj.shape)

        return torch.sparse.FloatTensor(indices, values, shape).to(self.device)

    def _sparse_dropout(
        self,
        x: torch.sparse.FloatTensor,
        dropout_rate: float
    ) -> torch.sparse.FloatTensor:
        """
        Apply dropout to sparse tensor by randomly dropping edges.

        Args:
            x: Input sparse tensor
            dropout_rate: Probability of dropping each edge

        Returns:
            Sparse tensor with dropout applied
        """
        if dropout_rate == 0.0:
            return x

        keep_prob = 1.0 - dropout_rate
        random_tensor = keep_prob + torch.rand(x._nnz()).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)

        indices = x._indices()[:, dropout_mask]
        values = x._values()[dropout_mask]

        out = torch.sparse.FloatTensor(indices, values, x.shape).to(x.device)
        return out * (1.0 / keep_prob)

    def forward(
        self,
        adjacency_matrix: torch.sparse.FloatTensor,
        users: torch.LongTensor,
        pos_items: torch.LongTensor,
        neg_items: torch.LongTensor,
        drop_flag: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the NGCF model.

        Args:
            adjacency_matrix: Normalized adjacency matrix of user-item graph
            users: User indices for the batch
            pos_items: Positive item indices for the batch
            neg_items: Negative item indices for the batch
            drop_flag: Whether to apply dropout

        Returns:
            Tuple of (user_embeddings, positive_item_embeddings, negative_item_embeddings)
        """
        # Apply node dropout to adjacency matrix
        if drop_flag and self.node_dropout > 0:
            A_hat = self._sparse_dropout(adjacency_matrix, self.node_dropout)
        else:
            A_hat = adjacency_matrix

        # Initial embeddings (ego-embeddings)
        ego_embeddings = torch.cat([
            self.embedding_dict['user_emb'],
            self.embedding_dict['item_emb']
        ], dim=0)

        # Store embeddings from all layers
        all_embeddings = [ego_embeddings]

        # Graph convolution layers
        for k in range(self.n_layers):
            # Message passing: aggregate neighbor information
            neighbor_embeddings = torch.sparse.mm(A_hat, ego_embeddings)

            # First-order propagation (sum of neighbors)
            sum_embeddings = torch.matmul(
                neighbor_embeddings,
                self.weight_dict[f'W_gc_{k}']
            ) + self.weight_dict[f'b_gc_{k}']

            # Bi-interaction (element-wise product with neighbors)
            bi_embeddings = torch.mul(ego_embeddings, neighbor_embeddings)
            bi_embeddings = torch.matmul(
                bi_embeddings,
                self.weight_dict[f'W_bi_{k}']
            ) + self.weight_dict[f'b_bi_{k}']

            # Combine and activate
            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(
                sum_embeddings + bi_embeddings
            )

            # Message dropout
            if drop_flag and self.message_dropout[k] > 0:
                ego_embeddings = F.dropout(
                    ego_embeddings,
                    p=self.message_dropout[k],
                    training=self.training
                )

            # Normalize embeddings
            ego_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings.append(ego_embeddings)

        # Concatenate all layer embeddings
        all_embeddings = torch.cat(all_embeddings, dim=1)

        # Split into user and item embeddings
        user_all_embeddings = all_embeddings[:self.n_users, :]
        item_all_embeddings = all_embeddings[self.n_users:, :]

        # Get embeddings for batch
        u_g_embeddings = user_all_embeddings[users]
        pos_i_g_embeddings = item_all_embeddings[pos_items]
        neg_i_g_embeddings = item_all_embeddings[neg_items]

        return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings

    def compute_bpr_loss(
        self,
        users_emb: torch.Tensor,
        pos_items_emb: torch.Tensor,
        neg_items_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Bayesian Personalized Ranking (BPR) loss.

        BPR loss encourages positive items to be ranked higher than negative items.
        Loss = -log(sigmoid(pos_score - neg_score)) + regularization

        Args:
            users_emb: User embeddings
            pos_items_emb: Positive item embeddings
            neg_items_emb: Negative item embeddings

        Returns:
            Tuple of (total_loss, bpr_loss, regularization_loss)
        """
        # Compute prediction scores
        pos_scores = torch.sum(users_emb * pos_items_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_items_emb, dim=1)

        # BPR loss
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # L2 regularization
        reg_loss = self.reg * (
            torch.norm(users_emb) ** 2 +
            torch.norm(pos_items_emb) ** 2 +
            torch.norm(neg_items_emb) ** 2
        ) / (2 * users_emb.shape[0])

        total_loss = bpr_loss + reg_loss

        return total_loss, bpr_loss, reg_loss

    def predict(
        self,
        adjacency_matrix: torch.sparse.FloatTensor,
        users: torch.LongTensor,
        items: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        """
        Generate prediction scores for user-item pairs.

        Args:
            adjacency_matrix: Normalized adjacency matrix
            users: User indices
            items: Item indices (if None, compute scores for all items)

        Returns:
            Prediction scores (batch_size, n_items) or (batch_size, len(items))
        """
        with torch.no_grad():
            # Get all embeddings without dropout
            ego_embeddings = torch.cat([
                self.embedding_dict['user_emb'],
                self.embedding_dict['item_emb']
            ], dim=0)

            all_embeddings = [ego_embeddings]

            for k in range(self.n_layers):
                neighbor_embeddings = torch.sparse.mm(adjacency_matrix, ego_embeddings)

                sum_embeddings = torch.matmul(
                    neighbor_embeddings,
                    self.weight_dict[f'W_gc_{k}']
                ) + self.weight_dict[f'b_gc_{k}']

                bi_embeddings = torch.mul(ego_embeddings, neighbor_embeddings)
                bi_embeddings = torch.matmul(
                    bi_embeddings,
                    self.weight_dict[f'W_bi_{k}']
                ) + self.weight_dict[f'b_bi_{k}']

                ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(
                    sum_embeddings + bi_embeddings
                )
                ego_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
                all_embeddings.append(ego_embeddings)

            all_embeddings = torch.cat(all_embeddings, dim=1)
            user_embeddings = all_embeddings[:self.n_users, :]
            item_embeddings = all_embeddings[self.n_users:, :]

            # Get user embeddings for batch
            u_emb = user_embeddings[users]

            # Compute scores
            if items is None:
                # All items
                scores = torch.matmul(u_emb, item_embeddings.t())
            else:
                # Specific items
                i_emb = item_embeddings[items]
                scores = torch.matmul(u_emb, i_emb.t())

            return scores

    def recommend(
        self,
        adjacency_matrix: torch.sparse.FloatTensor,
        user_id: int,
        k: int = 10,
        exclude_seen: bool = True,
        seen_items: Optional[set] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate top-k recommendations for a single user.

        Args:
            adjacency_matrix: Normalized adjacency matrix
            user_id: User ID
            k: Number of recommendations
            exclude_seen: Whether to exclude items the user has already seen
            seen_items: Set of item IDs the user has interacted with

        Returns:
            Tuple of (recommended_item_ids, prediction_scores)
        """
        user_tensor = torch.LongTensor([user_id]).to(self.device)
        scores = self.predict(adjacency_matrix, user_tensor).squeeze()

        # Exclude seen items
        if exclude_seen and seen_items:
            scores[list(seen_items)] = -np.inf

        # Get top-k items
        top_k_scores, top_k_items = torch.topk(scores, k)

        return top_k_items.cpu().numpy(), top_k_scores.cpu().numpy()


def save_model(model: NGCF, path: str) -> None:
    """Save model checkpoint."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_users': model.n_users,
        'n_items': model.n_items,
        'embedding_size': model.embedding_size,
        'layer_sizes': model.layer_sizes,
        'node_dropout': model.node_dropout,
        'message_dropout': model.message_dropout,
        'regularization': model.reg
    }, path)


def load_model(path: str, device: str = 'cuda') -> NGCF:
    """Load model from checkpoint."""
    checkpoint = torch.load(path, map_location=device)

    model = NGCF(
        n_users=checkpoint['n_users'],
        n_items=checkpoint['n_items'],
        embedding_size=checkpoint['embedding_size'],
        layer_sizes=checkpoint['layer_sizes'],
        node_dropout=checkpoint['node_dropout'],
        message_dropout=checkpoint['message_dropout'],
        regularization=checkpoint['regularization'],
        device=device
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    return model
