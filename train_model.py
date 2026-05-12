"""
train_model.py
Trains K-Means and PCA models, finds optimal K, and saves artifacts.
"""
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

# STEP 1 — Define features for clustering
CLUSTER_FEATURES = [
    'annual_income',       # Economic power
    'spending_score',      # Spending willingness
    'recency_days',        # How recently they bought
    'frequency',           # How often they buy
    'monetary',            # How much they spend
    'avg_order_value',     # Basket size
    'online_purchase_ratio', # Digital preference
    'loyalty_years',       # Relationship length
    'discount_usage_rate', # Price sensitivity
    'returns_rate',        # Satisfaction proxy
    'support_tickets',     # Service usage
    'clv_score',           # CLV proxy
    'engagement_score',    # Overall engagement
]

def main() -> None:
    print("Loading data...")
    # STEP 2 — Load and scale
    try:
        df = pd.read_csv("data/customers.csv")
    except FileNotFoundError:
        print("Error: data/customers.csv not found. Run generate_data.py first.")
        return

    X = df[CLUSTER_FEATURES].copy()
    
    # Check for NaN — fill with column median if any exist
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.median())
        
    print(f"Feature matrix shape: {X.shape}")
    print("\nFeature stats:")
    print(X.describe().loc[['min', 'max', 'mean']].round(2))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    # STEP 3 — PCA for visualization (2 components)
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    joblib.dump(pca_2d, "models/pca_model.pkl")
    print(f"\nPCA explained variance: {pca_2d.explained_variance_ratio_}")
    print(f"Total variance captured: {pca_2d.explained_variance_ratio_.sum()*100:.1f}%")
    
    # Compute PCA with n_components=min(13, n_samples)
    pca_full = PCA(n_components=min(len(CLUSTER_FEATURES), len(X)), random_state=42)
    pca_full.fit(X_scaled)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    print("Cumulative explained variance:")
    for i, cum_var in enumerate(cumulative_variance):
        print(f"  {i+1} components: {cum_var*100:.1f}%")
        
    # STEP 4 — Find Optimal K using Elbow Method + Silhouette Score
    k_range = range(2, 11)
    inertias, silhouette_scores, db_scores = [], [], []

    print("\nTesting K values from 2 to 10...")
    print(f"{'K':>3} | {'Inertia':>12} | {'Silhouette':>12} | {'Davies-Bouldin':>15}")
    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=20,
                    max_iter=500, random_state=42)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
        db_scores.append(davies_bouldin_score(X_scaled, labels))
        print(f"{k:>3} | {km.inertia_:>12,.0f} | {silhouette_scores[-1]:>12.4f} | {db_scores[-1]:>15.4f}")

    best_k_silhouette = k_range[np.argmax(silhouette_scores)]
    print(f"\nSilhouette recommends K={best_k_silhouette}")
    print("Selected K=5 to match known archetype count [OK]")
    
    # STEP 5 — Train final K-Means with K=5
    OPTIMAL_K = 5
    kmeans = KMeans(
        n_clusters=OPTIMAL_K,
        init='k-means++',     # Smart centroid initialization
        n_init=50,            # Run 50 times, keep best result
        max_iter=1000,        # Maximum iterations per run
        tol=1e-6,             # Convergence tolerance
        random_state=42
    )
    cluster_labels = kmeans.fit_predict(X_scaled)
    joblib.dump(kmeans, "models/kmeans_model.pkl")

    # STEP 6 — Evaluate final model
    final_sil = silhouette_score(X_scaled, cluster_labels)
    final_db = davies_bouldin_score(X_scaled, cluster_labels)
    print(f"\nFinal K-Means: Inertia={kmeans.inertia_:,.0f} | Silhouette={final_sil:.3f} | DB={final_db:.3f}")
    
    print("\nCluster sizes:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Cluster {u}: {c}")

    # STEP 7 — Build segment profiles
    df['cluster'] = cluster_labels
    cluster_profiles = df.groupby('cluster')[CLUSTER_FEATURES].mean()
    
    # Assign names dynamically based on distinct feature characteristics
    cluster_monetary = cluster_profiles['monetary'].sort_values(ascending=False).index.tolist()
    premium_cluster = cluster_monetary[0]
    
    cluster_recency = cluster_profiles['recency_days'].sort_values(ascending=False).index.tolist()
    at_risk_cluster = next(c for c in cluster_recency if c != premium_cluster)
    
    cluster_discount = cluster_profiles['discount_usage_rate'].sort_values(ascending=False).index.tolist()
    bargain_cluster = next(c for c in cluster_discount if c not in [premium_cluster, at_risk_cluster])
    
    cluster_online = cluster_profiles['online_purchase_ratio'].sort_values(ascending=False).index.tolist()
    young_cluster = next(c for c in cluster_online if c not in [premium_cluster, at_risk_cluster, bargain_cluster])
    
    occasional_cluster = next(c for c in range(5) if c not in [premium_cluster, at_risk_cluster, bargain_cluster, young_cluster])
    
    SEGMENT_NAMES = {
        premium_cluster: "Premium Loyalists",
        at_risk_cluster: "At-Risk High-Value",
        bargain_cluster: "Bargain Hunters",
        young_cluster: "Young Explorers",
        occasional_cluster: "Occasional Shoppers"
    }
    
    print("\nAssigned Segment Names:")
    for c, name in SEGMENT_NAMES.items():
        print(f"  Cluster {c}: {name}")
    
    profiles = {}
    for c in range(OPTIMAL_K):
        subset = df[df['cluster'] == c]
        profile = {
            'size': len(subset),
            'pct': len(subset) / len(df),
            'dominant_product_category': subset['product_category_preference'].mode()[0],
            'dominant_gender': subset['gender'].mode()[0],
            'avg_age': subset['age'].mean(),
            'name': SEGMENT_NAMES[c]
        }
        for feat in CLUSTER_FEATURES:
            profile[f'{feat}_mean'] = subset[feat].mean()
            
        profile['annual_income_median'] = subset['annual_income'].median()
        profile['monetary_median'] = subset['monetary'].median()
        profile['frequency_median'] = subset['frequency'].median()
        
        profiles[c] = profile

    joblib.dump(profiles, "models/segment_profiles.pkl")
    
    # STEP 8 — Add cluster labels and PCA coords to dataset
    df['segment_name'] = df['cluster'].map(SEGMENT_NAMES)
    df['pca_x'] = X_pca_2d[:, 0]
    df['pca_y'] = X_pca_2d[:, 1]
    df.to_csv("data/customers_segmented.csv", index=False)
    
    # STEP 9 — Print full summary
    print("\nAll models saved [OK]")
    
    # STEP 10 — Compare with DBSCAN
    dbscan = DBSCAN(eps=0.8, min_samples=10, n_jobs=-1)
    db_labels = dbscan.fit_predict(X_scaled)
    n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise = list(db_labels).count(-1)
    print(f"\nDBSCAN comparison: {n_clusters_db} clusters, {n_noise} noise points")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during training: {e}")
