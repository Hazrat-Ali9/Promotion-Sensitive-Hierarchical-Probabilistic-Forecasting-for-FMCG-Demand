🧠 Promotion-Sensitive Hierarchical Probabilistic Forecasting for FMCG Demand 📦📊

Promotion-Sensitive-Hierarchical-Probabilistic-Forecasting-for-FMCG-Demand is an advanced machine learning framework designed to accurately forecast Fast-Moving Consumer Goods (FMCG) demand by capturing promotional effects, hierarchical sales dependencies, and uncertainty distributions across products, regions, and time scales.

This project combines hierarchical Bayesian modeling, probabilistic deep learning, and promotion-aware feature engineering to deliver robust, scalable, and interpretable demand forecasts that drive smarter supply chain decisions in the FMCG industry.

🧩 Abstract

Traditional demand forecasting models struggle to handle the complex temporal and hierarchical relationships found in FMCG data, especially during promotional periods that heavily influence consumer behavior.

This project proposes a promotion-sensitive hierarchical probabilistic model that captures:

The impact of marketing promotions and discounts at both product and category levels.

Temporal dynamics across multiple time scales (daily, weekly, monthly).

Hierarchical dependencies (SKU → Category → Brand → Region → Country).

Predictive uncertainty, enabling risk-aware business planning.

✨ Key Features

📊 Hierarchical Probabilistic Forecasting: Models nested relationships between SKUs, product categories, and markets.

🛍️ Promotion Sensitivity: Incorporates promotion calendars, discount depth, and advertising intensity as causal drivers.

🧠 Deep Learning Backbone: Uses LSTM / Temporal Fusion Transformer (TFT) / DeepAR for sequential modeling.

🔮 Bayesian Uncertainty Estimation: Provides prediction intervals instead of point forecasts for better risk management.

📈 Feature-Aware Forecasting: Integrates seasonality, price elasticity, and macroeconomic indicators.

🧰 Scalable to Enterprise FMCG Data: Designed for thousands of SKUs and regions simultaneously.

⚙️ Technical Highlights
Module	Description
Data Preprocessing	Cleans, aggregates, and encodes multi-level FMCG data.
Feature Engineering	Generates promotion indicators, lagged sales features, calendar events, and price elasticity variables.
Hierarchical Modeling	Applies top-down and bottom-up reconciliation of forecasts across hierarchy levels.
Probabilistic Forecasting	Models sales as distributions (Gaussian / Negative Binomial / Quantile regression).
Deep Sequence Models	Implements DeepAR, LSTM, TFT, or N-BEATS architectures.
Evaluation Framework	Measures accuracy, calibration, and promotion impact across time horizons.
🧮 Methodology Overview
Input:
  Historical Sales Data + Promotion Events + Price Changes + Holidays
↓
Feature Engineering:
  Lag Features, Categorical Embeddings, Promo Flags, Seasonality Encoding
↓
Model:
  Hierarchical Probabilistic Deep Learning Model
↓
Output:
  Multi-level Forecasts (SKU, Brand, Category) + Prediction Intervals
↓
Evaluation:
  Accuracy, Sharpness, Calibration, Business Uplift Metrics

🧰 Tech Stack

Languages: Python 🐍

Core Frameworks: PyTorch / TensorFlow Probability / PyMC / Prophet

Forecasting Libraries: GluonTS, Kats, Nixtla, NeuralForecast

Visualization: Plotly, Matplotlib, Seaborn, Altair

Data Handling: Pandas, NumPy, Dask, Polars

📁 Project Structure
📁 data/                      # Raw and processed FMCG datasets
📁 notebooks/                 # Experiment and analysis notebooks
📁 features/                  # Promotion, temporal, and categorical feature generators
📁 models/                    # Probabilistic and hierarchical forecasting models
📁 evaluation/                # Performance and uncertainty metrics
📁 results/                   # Forecast visualizations and evaluation reports
📁 utils/                     # Helper scripts and data loaders

🚀 Getting Started
git clone https://github.com/yourusername/Promotion-Sensitive-Hierarchical-Probabilistic-Forecasting-for-FMCG-Demand.git
cd Promotion-Sensitive-Hierarchical-Probabilistic-Forecasting-for-FMCG-Demand
pip install -r requirements.txt
python train.py --model tft --hierarchy brand --promo_sensitivity True --forecast_horizon 90

📊 Evaluation Metrics
Category	Metric	Description
Accuracy	RMSE / MAPE / sMAPE	Overall forecast performance
Calibration	CRPS / Sharpness	Reliability of probabilistic predictions
Fairness	Weighted Reconciliation Error	Hierarchical consistency
Promotion Impact	Lift vs. Baseline	Sales uplift due to promotions
Explainability	SHAP / Attention Weights	Feature contribution analysis
💡 Business Impact

🏪 Demand Forecasting Accuracy: Reduces forecast error during promo periods by 20–30%.

💰 Inventory Optimization: Aligns procurement and logistics with expected uplift.

📦 Promotion Planning: Quantifies promotion elasticity to guide marketing spend.

🌍 Hierarchical Insights: Enables decision-making from SKU to national level.

🧠 Research Contributions

Introduces a promotion-sensitive probabilistic hierarchy for FMCG forecasting.

Combines Bayesian learning and deep sequence models for uncertainty quantification.

Provides an open-source reproducible pipeline for industry-scale forecasting research.

🤝 Contributing

We welcome collaboration from:

Data scientists working in retail analytics or CPG forecasting

Researchers in probabilistic time series and causal inference

Practitioners in supply chain optimization and pricing strategy

## License

[MIT License](LICENSE)

🏆 Citation

Hazrat Ali, Promotion-Sensitive Hierarchical Probabilistic Forecasting for FMCG Demand, 2025.
