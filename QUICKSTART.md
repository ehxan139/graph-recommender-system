# Quick Start Guide

This guide will help you set up and run the Graph-Based Recommendation System in under 10 minutes.

## Prerequisites

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- GPU optional (CPU works fine for small datasets)

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/ehxan139/graph-recommender-system.git
cd graph-recommender-system
```

### 2. Create Virtual Environment

```bash
# Create environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Demo

### Option 1: Train on Sample Data

```python
# demo.py
from src.models import LightGCN
from src.data.loader import RecommenderDataLoader
from src.evaluation.metrics import Trainer
import numpy as np
import scipy.sparse as sp

# Create sample data (100 users, 50 items, 500 interactions)
np.random.seed(42)
n_users, n_items = 100, 50
interactions = []

for _ in range(500):
    user = np.random.randint(0, n_users)
    item = np.random.randint(0, n_items)
    interactions.append({'user_id': user, 'item_id': item})

# Save to CSV
import pandas as pd
df = pd.DataFrame(interactions)
df.to_csv('sample_data.csv', index=False)

# Load data
loader = RecommenderDataLoader()
loader.load_from_csv('sample_data.csv')
loader.train_test_split(test_ratio=0.2)

print(f"Dataset: {loader.n_users} users, {loader.n_items} items")

# Train model
model = LightGCN(
    n_users=loader.n_users,
    n_items=loader.n_items,
    embedding_size=32,
    n_layers=2
)

trainer = Trainer(model, loader, learning_rate=0.001)
history = trainer.fit(epochs=20, eval_every=5, verbose=True)

# Generate recommendations
user_id = 0
recommendations, scores = model.recommend(
    trainer.adj_matrix,
    user_id=user_id,
    k=5
)

print(f"\nTop 5 recommendations for user {user_id}:")
for item, score in zip(recommendations, scores):
    print(f"  Item {item}: {score:.4f}")
```

Run the demo:
```bash
python demo.py
```

### Option 2: Start API Server

```bash
# Navigate to API directory
cd src/api

# Start server
uvicorn server:app --reload --port 8000
```

Visit: http://localhost:8000/docs for interactive API documentation

Test with curl:
```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "0",
    "k": 10,
    "model_name": "lightgcn"
  }'
```

## Using Your Own Data

### CSV Format

Your CSV should have at minimum:
- `user_id` column
- `item_id` column
- Optional: `rating` column (if explicit feedback)

Example:
```csv
user_id,item_id,rating,timestamp
user_001,item_123,4.5,1609459200
user_002,item_456,5.0,1609545600
```

Load your data:
```python
loader = RecommenderDataLoader(min_user_interactions=5)
loader.load_from_csv(
    'your_data.csv',
    user_col='user_id',
    item_col='item_id',
    rating_col='rating',
    rating_threshold=4.0  # Consider ratings >= 4.0 as positive
)
```

### JSON Format

For JSONL files (one JSON object per line):
```json
{"user_id": "user_001", "item_id": "item_123", "rating": 4.5}
{"user_id": "user_002", "item_id": "item_456", "rating": 5.0}
```

Load your data:
```python
loader.load_from_json(
    'your_data.jsonl',
    user_key='user_id',
    item_key='item_id'
)
```

## Common Issues

### Issue: Out of Memory

**Solution**: Reduce batch size or embedding size
```python
model = LightGCN(embedding_size=32, n_layers=2)  # Smaller model
trainer.fit(batch_size=512)  # Smaller batches
```

### Issue: Slow Training

**Solution**: Use GPU if available
```python
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LightGCN(..., device=device)
```

### Issue: Poor Recommendations

**Solution**: Tune hyperparameters
```python
model = LightGCN(
    embedding_size=64,  # Increase capacity
    n_layers=3,         # More graph hops
    reg_weight=1e-5     # Less regularization
)

trainer = Trainer(model, loader, learning_rate=0.001)
trainer.fit(epochs=100)  # Train longer
```

## Next Steps

1. **Explore Notebooks**: Check `notebooks/` for detailed examples
2. **Read Documentation**: See `docs/` for advanced features
3. **Deploy**: Use `deployment/` configs for production
4. **Customize**: Modify models in `src/models/` for your use case

## Getting Help

- **Issues**: Open an issue on GitHub
- **Documentation**: See README.md for comprehensive guide
- **API Docs**: http://localhost:8000/docs (when server running)

Happy recommending! 🚀
