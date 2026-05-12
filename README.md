# 🎯 Customer Segmentation Dashboard

An interactive, production-ready customer segmentation application powered by **K-Means Clustering**, **PCA dimensionality reduction**, and **Streamlit**.

## Features

- **5 Customer Segments** — Premium Loyalists, Occasional Shoppers, Bargain Hunters, At-Risk High-Value, Young Explorers
- **Interactive PCA Visualization** — 2D scatter plot of 13 behavioral features
- **RFM Analysis** — Recency, Frequency, Monetary scoring with 3D visualization
- **Customer Lookup** — Individual customer profiling and comparison
- **New Customer Prediction** — Real-time segment assignment for new customers
- **DBSCAN Outlier Detection** — Identify anomalous customer behavior
- **CRM Export** — Download segmented data and marketing action plans
- **Model Insights** — Elbow method, silhouette analysis, K-Means convergence animation

## Tech Stack

- **ML**: scikit-learn (K-Means, PCA, DBSCAN, StandardScaler)
- **Visualization**: Plotly (interactive charts, 3D plots, animations)
- **Frontend**: Streamlit (dark-themed dashboard)
- **Data**: 2,000 synthetic customers across 13 behavioral features

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python generate_data.py

# Train clustering model
python train_model.py

# Launch dashboard
streamlit run app.py
```

## Project Structure

```
├── app.py                  # Streamlit dashboard (main entry point)
├── segmentor.py            # ML pipeline, visualization functions
├── generate_data.py        # Synthetic data generation
├── train_model.py          # K-Means training pipeline
├── requirements.txt        # Python dependencies
├── data/
│   ├── customers.csv       # Raw customer data
│   └── customers_segmented.csv  # Segmented output
├── models/
│   ├── kmeans_model.pkl    # Trained K-Means model
│   ├── pca_model.pkl       # Fitted PCA transformer
│   ├── scaler.pkl          # Fitted StandardScaler
│   └── segment_profiles.pkl # Cluster profile summaries
└── .streamlit/
    └── config.toml         # Streamlit theme configuration
```

## License

MIT
