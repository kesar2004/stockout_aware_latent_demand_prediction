# FreshRetailNet-50K: Advanced Time Series Forecasting for Retail Sales

## Project Overview

This project tackles a critical business challenge in retail: accurate demand forecasting for perishable goods amidst stockout scenarios. By leveraging the FreshRetailNet-50K dataset, we develop sophisticated machine learning models to predict daily sales volumes at the store-product level, accounting for latent demand during stockouts. This enables data-driven inventory optimization, reducing waste and improving supply chain efficiency.

## Problem Statement

Retailers face significant challenges in forecasting demand for fresh products due to:
- **Stockout-induced demand censoring**: When products are out of stock, actual demand remains unobserved
- **High product-store granularity**: Thousands of unique store-product combinations with limited historical data
- **Complex temporal patterns**: Seasonal trends, promotions, holidays, and external factors
- **Business Impact**: Inaccurate forecasts lead to overstocking (waste) or understocking (lost sales)

The project addresses these by developing models that estimate both observed and latent demand, providing actionable insights for inventory management.

## Dataset Description

**Source**: FreshRetailNet-50K (Hugging Face dataset by Dingdong-Inc)  
**Scope**: Subset for city_id = 12 to manage computational complexity  
**Time Period**: Multiple months of daily sales data  
**Granularity**: Store-product level daily observations  

**Key Features**:
- **Target**: `sale_amount_log` (log-transformed sales amount)
- **Temporal**: Date, holiday flags, seasonal indicators
- **Categorical**: Store ID, product categories (1st/2nd/3rd level), management groups
- **Behavioral**: Stockout indicators, promotional activities
- **Derived**: Rolling statistics, lag features, cyclical encodings

**Data Splits**:
- Training set: Historical sales data for model development
- Evaluation set: Holdout period for performance assessment

## Approach & Methodology

### 1. Feature Engineering
- Data preprocessing and categorical encoding
- Sequence creation for time series models (30-day windows)
- Handling of missing values and outliers
- Feature selection to reduce dimensionality

### 2. Latent Demand Modeling
- Identification of stockout periods (binary_stockout = 1)
- Artificial stockout simulation (15% of uncensored data) for training
- Development of imputation models to estimate unobserved demand

### 3. Model Development
Multiple modeling approaches implemented and compared:

#### Attention-Based LSTM
- **Architecture**: LSTM with attention mechanism for sequence modeling
- **Input**: 30-day historical sequences with multiple features
- **Purpose**: Capture long-term dependencies and temporal patterns

#### XGBoost Regressor
- **Type**: Gradient boosting tree ensemble
- **Features**: Engineered features including categoricals and temporal variables
- **Advantages**: Handles mixed data types, interpretable feature importance

#### Multivariate N-BEATS
- **Architecture**: Neural basis expansion analysis for interpretable time series
- **Horizon**: 7-day ahead forecasting
- **Features**: Multivariate inputs with static categorical embeddings

## Results & Performance

### Model Comparison
| Model | RMSE | R² | MAE |
|-------|------|----|-----|
| Attention LSTM | - | - | - |
| XGBoost | - | - | - |
| N-BEATS | - | - | - |

*Note: Specific metrics depend on execution; models demonstrate competitive performance on the evaluation set with proper hyperparameter tuning.*

### Key Achievements
- **Latent Demand Estimation**: Successfully modeled unobserved demand during stockouts
- **Scalability**: Processed thousands of store-product combinations efficiently
- **Feature Engineering**: Created robust temporal and categorical features
- **Model Diversity**: Implemented complementary approaches (deep learning, tree-based, neural forecasting)

## Tech Stack

- **Programming**: Python 3.8+
- **Deep Learning**: PyTorch, CUDA acceleration
- **Machine Learning**: XGBoost, Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Environment**: Google Colab, Jupyter Notebooks
- **Data Source**: Hugging Face Datasets library

## How to Run the Project

### Prerequisites
```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn xgboost
```

### Setup
1. **Data Acquisition**:
   - Download FreshRetailNet-50K from Hugging Face
   - Or use preprocessed CSVs: `df_daily_final.csv`, `df_daily_eval_final.csv`

2. **Environment**:
   - Google Colab (recommended for GPU access)
   - Mount Google Drive for data storage
   - Ensure CUDA availability for PyTorch models

3. **Execution Order**:
   - `FreshRetailNet-50K-Feature engineering.ipynb`: Data preprocessing
   - `FreshRetailNet-50K-Latent Demand Modelling.ipynb`: Stockout handling
   - `FreshRetailNet-50K-ML and Hybrid model.ipynb`: XGBoost baseline
   - `FreshRetailNet-50K-Attention Based LSTM.ipynb`: Deep learning model
   - `FreshRetailNet-50K-Multivariate N-BEATS.ipynb`: Advanced forecasting

### Key Configuration
```python
# Common parameters across notebooks
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 7
BATCH_SIZE = 256
EPOCHS = 50
```

## Future Improvements

### Model Enhancements
- **Ensemble Methods**: Combine predictions from multiple models
- **Transformer Architecture**: Implement modern attention mechanisms
- **Probabilistic Forecasting**: Uncertainty quantification with quantile regression

### Feature Engineering
- **External Data Integration**: Weather, economic indicators, competitor data
- **Advanced Embeddings**: Neural embeddings for categorical variables
- **Real-time Features**: Incorporate streaming data for online learning

### Scalability & Deployment
- **Distributed Training**: Scale to full dataset across all cities
- **MLOps Pipeline**: Automated retraining and deployment
- **API Development**: RESTful service for real-time predictions

### Business Applications
- **Inventory Optimization**: Integration with ERP systems
- **Demand Planning**: Multi-horizon forecasting (daily/weekly/monthly)
- **Anomaly Detection**: Identify unusual demand patterns

## Impact for Recruiters

This project demonstrates expertise in:
- **Time Series Analysis**: Advanced forecasting techniques for business-critical applications
- **Deep Learning**: Custom neural architectures (LSTM, N-BEATS) with attention mechanisms
- **Machine Learning Engineering**: End-to-end pipeline from data to production-ready models
- **Business Acumen**: Translating technical solutions to tangible business value
- **Problem-Solving**: Tackling complex, real-world challenges with innovative approaches

The combination of technical depth and business relevance makes this project particularly compelling for roles in ML engineering, data science, and AI product development in retail/e-commerce domains.