# Project: Data Processing and Analysis for Health Public Agency
## Open Food Facts Dataset Analysis & Imputation Feasibility

[![Docker](https://img.shields.io/badge/Docker-24.0+-blue.svg)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.12+-yellow.svg)](https://www.python.org/)

###  Project Context
This project was developed for the French Health Public Agency to enhance the Open Food Facts database. The mission involves cleaning the dataset and evaluating the feasibility of an auto-completion system for missing nutritional values.

###  Business & Technical Objectives
- **Data Industrialization**: Clean and standardize a massive dataset of food products.
- **Nutritional Analysis**: Explore the relationships between different nutrients and the Nutri-Score.
- **Imputation Strategy**: Evaluate if missing nutritional values can be accurately predicted using other product attributes (category, ingredients, etc.).

###  Technical Architecture
1. **Preprocessing Pipeline**: Automated cleaning and outlier detection.
2. **Statistical Analysis**: Correlation analysis and PCA for multivariate exploration.
3. **Visualization**: Interactive dashboards using Plotly and Matplotlib.
4. **Caching System**: Optimized data loading for large datasets.

---

###  Quick Start (Docker)

#### 1. Prerequisites
- Docker Desktop
- Docker Compose V2

#### 2. Launch the System
```bash
docker-compose up --build
```

#### 3. Access the Services
- **Jupyter Notebook**: [http://localhost:8883](http://localhost:8883) (Open mission3.ipynb)

---

###  Project Structure
```text
 mission3.ipynb       # Main analysis notebook
 src/
    pipeline/        # Imputation and cleaning pipelines
    scripts/         # Analysis and visualization scripts
    utils/           # Data loading and caching utilities
 dataset/             # Open Food Facts CSV data
 docker-compose.yml   # Container orchestration
 Dockerfile           # Python environment
```

###  Key Insights
- **Nutrient Correlations**: Strong correlations exist between specific nutrients (e.g., sugars and carbohydrates), which can be leveraged for data validation and imputation.
- **Category-Based Patterns**: Food categories (PNNS groups) are the strongest predictors of nutritional profiles.
- **Data Quality**: While the dataset is vast, high missingness in specific fields requires a sophisticated multi-step cleaning and imputation pipeline.

---
*This project demonstrates expertise in large-scale data cleaning, statistical analysis, and the development of data-driven suggestions for public health databases.*
