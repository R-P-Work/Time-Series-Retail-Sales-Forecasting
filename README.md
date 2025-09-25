# Sales Forecasting with Time Series & Machine Learning

This project applies modern **time series forecasting** and **machine learning** techniques to predict future product sales for a large retail dataset. It was originally developed as a project for the *Advances in Data Mining* course at Leiden University.

## Project Overview
- **Goal:** Predict total sales for each product and store for the upcoming month.  
- **Dataset:** ~2.9M sales records covering **34 months** (Jan 2013 – Oct 2015).  
- **Scope:** Forecast sales for ~22k unique items across 60 shops and 84 categories.  
- **Primary Evaluation Metric:** Root Mean Squared Error (RMSE).

## Exploratory Data Analysis & Preprocessing
Before modeling, we performed extensive exploratory data analysis (EDA) and preprocessing, including:
- **Data quality checks:** Verified there were no missing values, identified and removed one product with invalid negative pricing.  
- **Handling negative sales counts:** Interpreted as product returns and kept them in the dataset.  
- **Trend and seasonality analysis:** Identified strong sales peaks around Christmas.  
- **Shop and category insights:** Found that only a small number of categories and items drive most of the sales (power-law distribution).  
- **Data preparation:** Removed anomalies, aggregated sales by month, and structured features for model input.  

This step was essential to ensure that downstream models could capture both **seasonal patterns** and **shop/item-level variability** effectively.


## Methods & Models
We explored multiple forecasting approaches:

1. **Prophet Algorithm (Created by Meta)**  
   - Designed for seasonal/trend-heavy time series.  
   - Strong at detecting trend change points and holiday effects.  
   - Limitation: item-by-item prediction required, making it computationally expensive.

2. **Holt-Winters Exponential Smoothing (HWES)**  
   - Captures trend + seasonality with weighted historical averages.  
   - Tested multiple architectures (additive vs multiplicative).  
   - Additive trend + seasonality performed best.

3. **eXtreme Gradient Boosting Regressor (XGBRegressor)**  
   - Gradient boosting model optimized for speed and accuracy.  
   - Tuned hyperparameters (depth, learning rate, subsampling).  
   - Achieved best performance for Kaggle competition.

## Results
- **Prophet:** RMSE ~16,928 (aggregate predictions).  
- **HWES (best config):** RMSE ~13,781.  
- **XGBoost:**  
  - Default params → Accuracy: 0.68 | MSE: 47.19  
  - Tuned params → Accuracy: 0.886 | MSE: 18.71 | RMSE: 4.325  
- **Final Kaggle Submission (Group75):** RMSE = **1.05171** (slightly better than competition median)

## Tech Stack
- **Languages:** Python  
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels, xgboost, prophet  
- **Tools:** Jupyter Notebook, Kaggle Notebooks  

## Visualizations
- Seasonal trends in sales (holiday spikes).  
- Shop/category/item performance distributions.  
- Price and frequency distributions (power-law behavior).  
- Forecast curves for Prophet, HWES, and XGBoost.  

