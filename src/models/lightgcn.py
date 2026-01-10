"""
LightGCN: Simplified Graph Convolution Network for Recommendation

Production-ready implementation of LightGCN, a simplified and more effective
variant of graph convolutional networks for collaborative filtering.

Key Improvements over NGCF:
- Removes feature transformation and nonlinear activation
- Simpler model with fewer parameters
- Often achieves better performance with faster training
- More stable and easier to tune

Reference:
He et al. "LightGCN: Simplifying and Powering Graph Convolution Network
for Recommendation" (SIGIR 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import scipy.sparse as sp


class LightGCN(nn.Module):
    """
    LightGCN model for collaborative filtering.
    
    Simplifies the design of GCN for recommendation by removing:
    - Feature transformation matrices
    - Nonlinear activation functions
    
    Keeps only the most essential component: neighborhood aggregation
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_size: int = 64,
        n_layers: int = 3,
        alpha: Optional[list] = None,
        reg_weight: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize LightGCN model.
        
        Args:
            n_users: Number of users
            n_items: Number of items
            embedding_size: Dimension of embeddings
            n_layers: Number of graph convolution layers
            alpha: Layer combination weights (if None, uses uniform weights)
            reg_weight: L2 regularization weight
            device: Device to run on ('cuda' or 'cpu')
        """
        super(LightGCN, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.reg_weight = reg_weight
        self.device = device
        
        # Layer combination weights (default: uniform)
        if alpha is None:
            self.alpha = [1.0 / (n_layers + 1)] * (n_layers + 1)
        else:
            self.alpha = alpha
            
        # Initialize embeddings only (no weight matrices needed)
        self.user_embedding = nn.Embedding(n_users, embedding_size)
        self.item_embedding = nn.Embedding(n_items, embedding_size)
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
    def create_adjacency_matrix(
        self,
        user_item_matrix: sp.csr_matrix
    ) -> torch.sparse.FloatTensor:
        """
        Create normalized adjacency matrix for user-item bipartite graph.
        
        Applies symmetric normalization: D^(-1/2) * A * D^(-1/2)
        
        Args:
            user_item_matrix: User-item interaction matrix (sparse)
            
        Returns:
            Normalized sparse adjacency tensor
        """
        # Build adjacency matrix [[0, R], [R^T, 0]]
        n_nodes = self.n_users + self.n_items
        adj_mat = sp.dok_matrix((n_nodes, n_nodes), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        
        R = user_item_matrix.tocoo()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        
        # Compute D^(-1/2)
        rowsum = np.array(adj_mat.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        # Symmetric normalization
        norm_adj = d_mat_inv_sqrt @ adj_mat @ d_mat_inv_sqrt
        norm_adj = norm_adj.tocoo()
        
        # Convert to PyTorch sparse tensor
        indices = torch.LongTensor(np.vstack([norm_adj.row, norm_adj.col]))
        values = torch.FloatTensor(norm_adj.data)
        shape = torch.Size(norm_adj.shape)
        
        return torch.sparse.FloatTensor(indices, values, shape).to(self.device)
    
    def graph_convolution(
        self,
        adjacency_matrix: torch.sparse.FloatTensor
    ) -> torch.Tensor:
        """
        Perform graph convolution through multiple layers.
        
        Simply propagates embeddings through the graph without any
        transformation or activation.
        
        Args:
            adjacency_matrix: Normalized adjacency matrix
            
        Returns:
            Combined embeddings from all layers
        """
        # Initial embeddings (layer 0)
        ego_embeddings = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)
        
        # Store embeddings from each layer
        all_embeddings = [ego_embeddings]
        
        # Propagate through layers
        for layer in range(self.n_layers):
            # Simple neighbor aggregation: A * E
            ego_embeddings = torch.sparse.mm(adjacency_matrix, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        
        # Weighted combination of all layers
        final_embeddings = torch.stack(all_embeddings, dim=1)
        alpha_tensor = torch.tensor(self.alpha, dtype=torch.float32, device=self.device)
        final_embeddings = torch.sum(
            final_embeddings * alpha_tensor.view(1, -1, 1),
            dim=1
        )
        
        # Split back into user and item embeddings
        user_emb = final_embeddings[:self.n_users, :]
        item_emb = final_embeddings[self.n_users:, :]
        
        return user_emb, item_emb
    
    def forward(
        self,
        adjacency_matrix: torch.sparse.FloatTensor,
        users: torch.LongTensor,
        pos_items: torch.LongTensor,
        neg_items: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through LightGCN.
        
        Args:
            adjacency_matrix: Normalized adjacency matrix
            users: User indices
            pos_items: Positive item indices
            neg_items: Negative item indices
            
        Returns:
            Tuple of (user_emb, pos_item_emb, neg_item_emb,
                     user_emb_ego, pos_item_emb_ego, neg_item_emb_ego)
            where *_ego are the embeddings before graph convolution (for regularization)
        """
        # Get embeddings after graph convolution
        all_users_emb, all_items_emb = self.graph_convolution(adjacency_matrix)
        
        # Get embeddings for current batch
        users_emb = all_users_emb[users]
        pos_items_emb = all_items_emb[pos_items]
        neg_items_emb = all_items_emb[neg_items]
        
        # Get ego embeddings (layer 0, for regularization)
        users_emb_ego = self.user_embedding(users)
        pos_items_emb_ego = self.item_embedding(pos_items)
        neg_items_emb_ego = self.item_embedding(neg_items)
        
        return (users_emb, pos_items_emb, neg_items_emb,
                users_emb_ego, pos_items_emb_ego, neg_items_emb_ego)
    
    def compute_bpr_loss(
        self,
        users_emb: torch.Tensor,
        pos_items_emb: torch.Tensor,
        neg_items_emb: torch.Tensor,
        users_emb_ego: torch.Tensor,
        pos_items_emb_ego: torch.Tensor,
        neg_items_emb_ego: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute BPR loss with L2 regularization on ego embeddings.
        
        Args:
            users_emb: User embeddings after graph convolution
            pos_items_emb: Positive item embeddings after GC
            neg_items_emb: Negative item embeddings after GC
            users_emb_ego: Initial user embeddings (layer 0)
            pos_items_emb_ego: Initial positive item embeddings
            neg_items_emb_ego: Initial negative item embeddings
            
        Returns:
            Tuple of (total_loss, bpr_loss, reg_loss)
        """
        # Compute scores
        pos_scores = torch.sum(users_emb * pos_items_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_items_emb, dim=1)
        
        # BPR loss
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # L2 regularization on initial embeddings only
        reg_loss = self.reg_weight * (
            torch.norm(users_emb_ego) ** 2 +
            torch.norm(pos_items_emb_ego) ** 2 +
            torch.norm(neg_items_emb_ego) ** 2
        ) / (2 * users_emb_ego.shape[0])
        
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
            items: Item indices (if None, score all items)
            
        Returns:
            Prediction scores
        """
        with torch.no_grad():
            all_users_emb, all_items_emb = self.graph_convolution(adjacency_matrix)
            users_emb = all_users_emb[users]
            
            if items is None:
                # Score all items
                scores = torch.matmul(users_emb, all_items_emb.t())
            else:
                # Score specific items
                items_emb = all_items_emb[items]
                scores = torch.matmul(users_emb, items_emb.t())
            
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
        Generate top-k recommendations for a user.
        
        Args:
            adjacency_matrix: Normalized adjacency matrix
            user_id: User ID
            k: Number of recommendations
            exclude_seen: Whether to exclude seen items
            seen_items: Set of item IDs user has interacted with
            
        Returns:
            Tuple of (recommended_item_ids, scores)
        """
        user_tensor = torch.LongTensor([user_id]).to(self.device)
        scores = self.predict(adjacency_matrix, user_tensor).squeeze()
        
        # Exclude seen items
        if exclude_seen and seen_items:
            scores[list(seen_items)] = -np.inf
        
        # Get top-k
        top_k_scores, top_k_items = torch.topk(scores, k)
        
        return top_k_items.cpu().numpy(), top_k_scores.cpu().numpy()
    
    def get_user_embedding(
        self,
        adjacency_matrix: torch.sparse.FloatTensor,
        user_id: int
    ) -> np.ndarray:
        """Get final embedding for a user."""
        with torch.no_grad():
            all_users_emb, _ = self.graph_convolution(adjacency_matrix)
            return all_users_emb[user_id].cpu().numpy()
    
    def get_item_embedding(
        self,
        adjacency_matrix: torch.sparse.FloatTensor,
        item_id: int
    ) -> np.ndarray:
        """Get final embedding for an item."""
        with torch.no_grad():
            _, all_items_emb = self.graph_convolution(adjacency_matrix)
            return all_items_emb[item_id].cpu().numpy()


def save_model(model: LightGCN, path: str) -> None:
    """Save LightGCN model checkpoint."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_users': model.n_users,
        'n_items': model.n_items,
        'embedding_size': model.embedding_size,
        'n_layers': model.n_layers,
        'alpha': model.alpha,
        'reg_weight': model.reg_weight
    }, path)


def load_model(path: str, device: str = 'cuda') -> LightGCN:
    """Load LightGCN model from checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    
    model = LightGCN(
        n_users=checkpoint['n_users'],
        n_items=checkpoint['n_items'],
        embedding_size=checkpoint['embedding_size'],
        n_layers=checkpoint['n_layers'],
        alpha=checkpoint['alpha'],
        reg_weight=checkpoint['reg_weight'],
        device=device
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
