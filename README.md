# Stockout-Aware Latent Demand Forecasting in Retail (FreshRetailNet-50K)

**Advanced demand forecasting system that handles censored demand caused by stockouts, processing 5,000+ store-product time series with ~13% improvement in WAPE through latent demand reconstruction.**

## 🚀 Key Achievements
- **Scalability**: Modeled 5,117 store–product time series using global forecasting
- **Stockout-Aware Modelling**: Reconstructed *latent demand* using CatBoost-based imputation
- **Performance Gain**: Reduced WAPE from 43.29% → 30.39% (~12.9% improvement)
- **Deep Learning**: Implemented Multivariate N-BEATS for global time series forecasting
- **Business Impact**: Enables accurate demand estimation → better inventory & fewer stockouts

## 📋 Problem Statement

Retail demand forecasting faces critical challenges with **censored demand** during stockouts:
- When products are out of stock, true demand remains unobserved
- Traditional models underestimate demand, leading to chronic understocking
- Fresh products have high waste costs from overstocking and lost sales from understocking
- Complex temporal patterns with holidays, promotions, and seasonality

This project develops stockout-aware forecasting models that estimate both observed and latent demand, providing accurate predictions for inventory optimization.

## 🔄 Approach

### End-to-End Pipeline
1. **Data Acquisition & Preprocessing**
   - Load FreshRetailNet-50K dataset from Hugging Face
   - Feature engineering: temporal features, categorical encoding, sequence creation

2. **Latent Demand Modeling**
   - Identify stockout periods using binary indicators
   - Simulate artificial stockouts (15% of uncensored data) for training
   - Develop imputation models to estimate unobserved demand

3. **Model Training & Validation**
   - Train multiple forecasting models on processed data
   - Cross-validation with time series splits
   - Hyperparameter tuning for optimal performance

4. **Evaluation & Deployment**
   - Compare models using WAPE, RMSE, MAE, R² metrics
   - Select best-performing model for production use

## 💡 Key Innovation

Unlike traditional forecasting models that use observed sales directly, this project:

- Treats stockouts as **censored observations**
- Reconstructs **true latent demand** using a CatBoost-based imputer
- Trains forecasting models on **unconstrained demand**

This shifts forecasting from:
➡️ *“What was sold”*  
to  
➡️ *“What customers actually wanted to buy”*


## 🤖 Models Used

- **Prophet**: Facebook's time series forecasting with trend, seasonality, and holiday components
- **CatBoost**: Gradient boosting on decision trees with categorical feature handling
- **N-BEATS**: Neural basis expansion analysis for interpretable time series forecasting
- **Attention-Based LSTM**: Deep learning sequence model with attention mechanism
- **Latent Demand Imputer**: Custom model to estimate demand during stockouts

## 📊 Dataset Description

- **Source**: FreshRetailNet-50K (Hugging Face)
- **Scope**: City subset (city_id = 12) with 5,117 store-product combinations
- **Timeframe**: Multiple months of daily sales data
- **Features**:
  - Target: `sale_amount_log` (log-transformed sales)
  - Temporal: dates, holidays, seasonal indicators
  - Categorical: store/product hierarchies, management groups
  - Behavioral: stockout flags, promotions, activity indicators
  - Derived: rolling statistics, lag features, cyclical encodings

## 📈 Key Results

| Model | WAPE | RMSE | MAE | R² |
|-------|------|------|-----|----|
| N-BEATS (Best) | **30.39%** | 0.8486 | 0.4839 | 0.7203 |
| Baseline Model | 47.42% | 1.3752 | 0.5451 | 0.2653 |
| **Improvement** | **-5.33%** | **-38.3%** | **-11.2%** | **+171.6%** |

- **12.9% reduction in WAPE** after correcting for stockout-induced demand censoring
- **38% reduction in RMSE** compared to baseline
- Successfully handled stockout scenarios with latent demand estimation
- Models scaled to 5,000+ time series without performance degradation

## 🛠 Tech Stack

- **Languages**: Python 3.8+
- **Deep Learning**: PyTorch, CUDA
- **ML Frameworks**: XGBoost, CatBoost, NeuralForecast
- **Time Series**: Prophet, N-BEATS
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Environment**: Google Colab, Jupyter Notebooks
- **Data**: Hugging Face Datasets

## 🚀 How to Run the Project

### Prerequisites
```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn xgboost catboost prophet neuralforecast
```

### Setup & Execution
1. **Clone Repository**
   ```bash
   git clone https://github.com/kesar2004/stockout_aware_latent_demand_prediction.git
   cd time-series-project
   ```

2. **Data Preparation**
   - Download FreshRetailNet-50K from Hugging Face
   - Or use preprocessed CSVs: `df_daily_final.csv`, `df_daily_eval_final.csv`

3. **Run Notebooks in Order**:
   - `FreshRetailNet-50K-Feature engineering.ipynb`
   - `FreshRetailNet-50K-Latent Demand Modelling.ipynb`
   - `FreshRetailNet-50K-ML and Hybrid model.ipynb`
   - `FreshRetailNet-50K-Attention Based LSTM.ipynb`
   - `FreshRetailNet-50K-Multivariate N-BEATS.ipynb`

4. **Configuration**
   ```python
   SEQUENCE_LENGTH = 30
   FORECAST_HORIZON = 7
   BATCH_SIZE = 256
   EPOCHS = 50
   ```

## 🔮 Future Improvements

- **Ensemble Methods**: Combine multiple models for better accuracy
- **Real-time Forecasting**: Online learning with streaming data
- **Probabilistic Forecasting**: Uncertainty quantification with prediction intervals
- **External Data Integration**: Weather, economic indicators, competitor data
- **MLOps Pipeline**: Automated retraining and model deployment
- **Scalability**: Distributed training for full dataset (all cities)
- **Business Integration**: ERP system integration for inventory optimization

---

🚀 This project demonstrates advanced skills in time series forecasting, deep learning, and solving real-world supply chain problems using data-driven methods.
