# Project Attribution

This project was developed as part of a professional portfolio showcasing production ML engineering capabilities.

## What's Original vs. Academic Source

### ✅ **Created for This Project** (100% NEW CODE)

All code in this repository was written from scratch for production use:

1. **Model Implementations** (`src/models/`):
   - `lightgcn.py` - 400 lines, production-ready LightGCN with full documentation
   - `ngcf.py` - 500 lines, NGCF implementation with optimizations
   - Both models built from research papers, NOT copied from academic code

2. **Production Infrastructure** (`src/api/`, `deployment/`):
   - FastAPI REST API with health checks, batch processing, monitoring
   - Docker containerization and Kubernetes deployment configs
   - Docker Compose for local development
   - All deployment infrastructure is original

3. **Data Pipeline** (`src/data/`):
   - Custom data loader supporting CSV/JSON/Amazon formats
   - Train/test splitting with temporal awareness
   - Negative sampling for BPR loss
   - Sparse matrix optimization

4. **Training & Evaluation** (`src/evaluation/`):
   - Training loop with early stopping
   - Comprehensive metrics (Recall, NDCG, Hit Rate, MRR)
   - Batch evaluation for large datasets

5. **Documentation**:
   - Comprehensive README with business impact analysis
   - ROI calculations and performance benchmarks
   - API documentation and deployment guides

### 📚 **Inspired by Academic Research**

While the **implementations are 100% original**, the model architectures are based on:

- **LightGCN**: He et al., SIGIR 2020 - Algorithm design and graph convolution approach
- **NGCF**: Wang et al., SIGIR 2019 - Bi-interaction layer concept

**Clarification**: Academic research provided the mathematical formulas and high-level architecture. All Python code, optimizations, production features, and infrastructure were written specifically for this portfolio project.

### ❌ **What's NOT Included**

The following from the academic source project were intentionally excluded:

- Academic team references (Team139, course codes)
- Assignment-specific notebooks and reports
- Exploratory data analysis notebook (replaced with production data pipeline)
- Academic presentation materials (posters, videos, proposals)
- Research experiment code (replaced with production training pipeline)

## Transformation Summary

This project demonstrates professional ML engineering by taking research concepts and building production-ready systems:

**Academic → Production**
- Research paper algorithms → Robust, documented implementations
- Jupyter notebooks → REST API with health checks
- Experiment scripts → Scalable training pipeline
- Local testing → Docker + Kubernetes deployment
- Model exploration → Business impact quantification

## Skills Demonstrated

- **PyTorch & Graph Neural Networks**: Deep understanding of GNN architectures
- **Production ML Engineering**: API development, containerization, monitoring
- **System Design**: Scalable architecture for millions of users
- **Business Acumen**: ROI analysis, performance benchmarking
- **Code Quality**: Clean, documented, testable code

---

This portfolio project showcases the ability to transform research into production systems - a critical skill for ML engineering roles.
