import os
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import silhouette_samples
from sklearn.cluster import DBSCAN, KMeans

# CONSTANTS
CLUSTER_FEATURES = [
    'annual_income', 'spending_score', 'recency_days', 'frequency',
    'monetary', 'avg_order_value', 'online_purchase_ratio',
    'loyalty_years', 'discount_usage_rate', 'returns_rate',
    'support_tickets', 'clv_score', 'engagement_score'
]

SEGMENT_NAMES = {
    0: "Premium Loyalists",
    1: "Occasional Shoppers",
    2: "Bargain Hunters",
    3: "At-Risk High-Value",
    4: "Young Explorers"
}

SEGMENT_COLORS = {
    "Premium Loyalists":    "#F39C12",
    "Occasional Shoppers":  "#3498DB",
    "Bargain Hunters":      "#2ECC71",
    "At-Risk High-Value":   "#E74C3C",
    "Young Explorers":      "#9B59B6"
}

SEGMENT_EMOJIS = {
    "Premium Loyalists":    "👑",
    "Occasional Shoppers":  "🛍️",
    "Bargain Hunters":      "🏷️",
    "At-Risk High-Value":   "⚠️",
    "Young Explorers":      "🚀"
}

SEGMENT_STRATEGIES = {
    "Premium Loyalists":
        "VIP program, exclusive early access, personal account manager, "
        "luxury product recommendations, anniversary rewards",
    "Occasional Shoppers":
        "Re-engagement campaigns, seasonal promotions, browse abandonment "
        "emails, convenience-focused messaging, free shipping offers",
    "Bargain Hunters":
        "Flash sale alerts, loyalty points program, bulk buy discounts, "
        "clearance section highlights, referral rewards",
    "At-Risk High-Value":
        "Win-back campaign, personal outreach call, premium upgrade offer, "
        "survey to identify pain points, exclusive return offer",
    "Young Explorers":
        "Social media engagement, trend-first notifications, student "
        "discounts, gamification, influencer collaborations"
}

FEATURE_DISPLAY = {
    'annual_income':          ('Annual Income ($)',       '$,.0f'),
    'spending_score':         ('Spending Score (1-100)',  '.1f'),
    'recency_days':           ('Days Since Last Purchase', '.0f'),
    'frequency':              ('Purchase Frequency/Year', '.1f'),
    'monetary':               ('Annual Spend ($)',         '$,.0f'),
    'avg_order_value':        ('Avg Order Value ($)',      '$,.2f'),
    'online_purchase_ratio':  ('Online Purchase Ratio',   '.1%'),
    'loyalty_years':          ('Loyalty Years',           '.1f'),
    'discount_usage_rate':    ('Discount Usage Rate',     '.1%'),
    'returns_rate':           ('Returns Rate',            '.1%'),
    'support_tickets':        ('Support Tickets/Year',    '.1f'),
    'clv_score':              ('CLV Score (proxy)',        ',.1f'),
    'engagement_score':       ('Engagement Score',        '.1f'),
}

# MODEL LOADING
try:
    kmeans = joblib.load("models/kmeans_model.pkl")
    pca = joblib.load("models/pca_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    profiles = joblib.load("models/segment_profiles.pkl")
except FileNotFoundError:
    pass # Expected during initial definition or before running train_model.py

# FUNCTIONS

def load_segmented_data() -> pd.DataFrame:
    """Loads pre-segmented customer data."""
    try:
        df = pd.read_csv("data/customers_segmented.csv")
        if 'segment_name' not in df.columns and 'cluster' in df.columns:
            df['segment_name'] = df['cluster'].map(SEGMENT_NAMES)
        return df
    except FileNotFoundError:
        raise FileNotFoundError("Segmented data not found. Run train_model.py first.")

def assign_segments(df_new: pd.DataFrame) -> pd.DataFrame:
    """Predicts segments for new raw customer data."""
    df = df_new.copy()
    
    # Engineer features if not present
    if 'clv_score' not in df.columns:
        df['clv_score'] = ((df['frequency'] * df['monetary']) / (df['recency_days'] + 1)).round(2)
    if 'engagement_score' not in df.columns:
        eng_score = (1 / df['recency_days'] * 100) + (df['frequency'] * 2) + \
                    (df.get('online_purchase_ratio', 0) * 20) - (df.get('support_tickets', 0) * 3)
        df['engagement_score'] = np.clip(eng_score, 0, 200).round(2)
    if 'avg_order_value' not in df.columns:
        df['avg_order_value'] = np.clip(df['monetary'] / df['frequency'], 1.0, None).round(2)
        
    # Align to CLUSTER_FEATURES
    for col in CLUSTER_FEATURES:
        if col not in df.columns:
            df[col] = 0
            
    X = df[CLUSTER_FEATURES].copy()
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    cluster_labels = kmeans.predict(X_scaled)
    
    # Add cluster and segment name
    df['cluster'] = cluster_labels
    df['segment_name'] = df['cluster'].map(SEGMENT_NAMES)
    
    # Add PCA coordinates
    X_pca = pca.transform(X_scaled)
    df['pca_x'] = X_pca[:, 0]
    df['pca_y'] = X_pca[:, 1]
    
    return df

def get_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a summary of each segment."""
    summary = []
    total_customers = len(df)
    
    for name in SEGMENT_NAMES.values():
        subset = df[df['segment_name'] == name]
        if len(subset) == 0:
            continue
            
        summary.append({
            'segment_name': name,
            'emoji': SEGMENT_EMOJIS.get(name, "🔘"),
            'count': len(subset),
            'pct_of_total': f"{(len(subset) / total_customers * 100):.1f}%",
            'avg_income': subset['annual_income'].mean(),
            'avg_monetary': subset['monetary'].mean(),
            'avg_frequency': subset['frequency'].mean(),
            'avg_recency': subset['recency_days'].mean(),
            'avg_spending_score': subset['spending_score'].mean(),
            'avg_loyalty_years': subset['loyalty_years'].mean(),
            'color': SEGMENT_COLORS.get(name, "#FFFFFF"),
            'strategy': SEGMENT_STRATEGIES.get(name, "")
        })
        
    return pd.DataFrame(summary)

def get_pca_scatter(df: pd.DataFrame, highlight_customer: str = None) -> go.Figure:
    fig = px.scatter(
        df, x='pca_x', y='pca_y', color='segment_name',
        color_discrete_map=SEGMENT_COLORS,
        hover_data=['customer_id', 'segment_name', 'annual_income', 'monetary', 'spending_score', 'frequency'],
        title="Customer Segments — PCA Visualization (2D)",
        labels={'pca_x': "PC1 (Economic Power + Monetary)", 'pca_y': "PC2 (Engagement + Recency)"},
        opacity=0.7, template='plotly_dark'
    )
    fig.update_traces(marker=dict(size=5))
    
    # Add centroids
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    fig.add_trace(go.Scatter(
        x=centroids_pca[:, 0], y=centroids_pca[:, 1],
        mode='markers', marker=dict(size=20, symbol='star', color='white', line=dict(width=1, color='black')),
        name='Centroids', hoverinfo='skip'
    ))
    
    if highlight_customer and 'customer_id' in df.columns:
        cust = df[df['customer_id'] == highlight_customer]
        if not cust.empty:
            fig.add_trace(go.Scatter(
                x=cust['pca_x'], y=cust['pca_y'],
                mode='markers', marker=dict(size=15, color='white', line=dict(width=2, color='black')),
                name=f"Customer: {highlight_customer}"
            ))
            fig.add_annotation(
                x=cust['pca_x'].iloc[0], y=cust['pca_y'].iloc[0],
                text=highlight_customer, showarrow=True, arrowhead=1
            )
            
    return fig

def get_radar_chart(segment_name: str) -> go.Figure:
    features = [
        'spending_score', 'frequency', 'monetary', 'loyalty_years',
        'online_purchase_ratio', 'engagement_score', 'recency_days', 'discount_usage_rate'
    ]
    
    global_df = load_segmented_data()
    segment_df = global_df[global_df['segment_name'] == segment_name]
    
    if segment_df.empty:
        return go.Figure()
        
    global_means = []
    segment_means = []
    
    for f in features:
        min_v = global_df[f].min()
        max_v = global_df[f].max()
        range_v = max_v - min_v if max_v != min_v else 1
        
        glob_m = (global_df[f].mean() - min_v) / range_v
        seg_m = (segment_df[f].mean() - min_v) / range_v
        
        if f == 'recency_days':
            glob_m = 1 - glob_m
            seg_m = 1 - seg_m
            
        global_means.append(glob_m)
        segment_means.append(seg_m)
        
    features.append(features[0])
    global_means.append(global_means[0])
    segment_means.append(segment_means[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=global_means, theta=features, fill=None,
        mode='lines', line=dict(color='gray', dash='dash'), name='Global Average'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=segment_means, theta=features, fill='toself',
        line=dict(color=SEGMENT_COLORS.get(segment_name, 'white')),
        name=segment_name
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f"{segment_name} — Feature Profile",
        template='plotly_dark', showlegend=True
    )
    return fig

def get_segment_comparison_bar(df: pd.DataFrame, feature: str) -> go.Figure:
    grouped = df.groupby('segment_name')[feature].agg(['mean', 'std']).reset_index()
    grouped = grouped.sort_values(by='mean', ascending=False)
    
    disp_name = FEATURE_DISPLAY.get(feature, (feature, ''))[0]
    
    fig = px.bar(
        grouped, x='segment_name', y='mean', error_y='std',
        color='segment_name', color_discrete_map=SEGMENT_COLORS,
        title=f"Segment Comparison — {disp_name}",
        template='plotly_dark'
    )
    fig.update_layout(xaxis_title="Segment", yaxis_title=disp_name)
    return fig

def get_rfm_3d_scatter(df: pd.DataFrame) -> go.Figure:
    df_plot = df.copy()
    min_i = df_plot['annual_income'].min()
    max_i = df_plot['annual_income'].max()
    # Handle division by zero
    range_i = max_i - min_i if max_i > min_i else 1
    df_plot['income_size'] = ((df_plot['annual_income'] - min_i) / range_i) * 20 + 5
    
    fig = px.scatter_3d(
        df_plot, x='recency_days', y='frequency', z='monetary',
        color='segment_name', color_discrete_map=SEGMENT_COLORS,
        size='income_size',
        hover_data=['customer_id', 'segment_name', 'recency_days', 'frequency', 'monetary'],
        title="RFM 3D Customer Map",
        template='plotly_dark'
    )
    return fig

def get_income_spend_scatter(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df, x='annual_income', y='spending_score', color='segment_name',
        color_discrete_map=SEGMENT_COLORS, marginal_x='histogram', marginal_y='histogram',
        hover_data=['customer_id', 'segment_name', 'monetary', 'frequency'],
        title="Annual Income vs Spending Score by Segment",
        template='plotly_dark'
    )
    return fig

def get_feature_distribution_box(df: pd.DataFrame, feature: str) -> go.Figure:
    disp_name = FEATURE_DISPLAY.get(feature, (feature, ''))[0]
    fig = px.box(
        df, x='segment_name', y=feature, color='segment_name',
        color_discrete_map=SEGMENT_COLORS, points='outliers',
        title=f"Distribution of {disp_name}",
        template='plotly_dark'
    )
    return fig

def get_segment_heatmap(df: pd.DataFrame) -> go.Figure:
    means = df.groupby('segment_name')[CLUSTER_FEATURES].mean()
    normalized = (means - means.min()) / (means.max() - means.min() + 1e-9)
    
    fig = px.imshow(
        normalized, color_continuous_scale='RdYlGn',
        aspect='auto',
        title="Segment Feature Profile Heatmap (Normalized)",
        template='plotly_dark'
    )
    fig.update_traces(text=means.round(1).values, texttemplate="%{text}")
    return fig

def get_elbow_chart(inertias: list, silhouettes: list) -> go.Figure:
    k_range = list(range(2, 2 + len(inertias)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k_range, y=inertias, name='Inertia', line=dict(color='blue'), yaxis='y1'))
    fig.add_trace(go.Scatter(x=k_range, y=silhouettes, name='Silhouette', line=dict(color='green'), yaxis='y2'))
    
    fig.update_layout(
        title="Elbow Method + Silhouette Score — Finding Optimal K",
        template='plotly_dark',
        yaxis=dict(title='Inertia', titlefont=dict(color='blue'), tickfont=dict(color='blue')),
        yaxis2=dict(title='Silhouette Score', titlefont=dict(color='green'), tickfont=dict(color='green'), anchor='x', overlaying='y', side='right'),
        xaxis=dict(title='Number of Clusters (K)'),
    )
    fig.add_vline(x=5, line_dash="dash", line_color="white", annotation_text="Optimal K=5", annotation_position="top right")
    return fig

def get_silhouette_plot(X_scaled, labels) -> go.Figure:
    scores = silhouette_samples(X_scaled, labels)
    mean_score = scores.mean()
    
    df_sil = pd.DataFrame({'score': scores, 'cluster': labels})
    df_sil.sort_values(['cluster', 'score'], ascending=[True, True], inplace=True)
    df_sil['y_pos'] = range(len(df_sil))
    
    fig = go.Figure()
    for c in sorted(df_sil['cluster'].unique()):
        subset = df_sil[df_sil['cluster'] == c]
        name = SEGMENT_NAMES.get(c, f"Cluster {c}")
        fig.add_trace(go.Bar(
            y=subset['y_pos'], x=subset['score'], orientation='h',
            name=name, marker_color=SEGMENT_COLORS.get(name, 'white'), marker_line_width=0
        ))
        
    fig.add_vline(x=mean_score, line_dash="dash", line_color="red", annotation_text="Mean Silhouette")
    fig.update_layout(
        title="Silhouette Analysis — K=5",
        template='plotly_dark',
        xaxis_title="Silhouette Coefficient",
        yaxis_title="Customers (sorted within each cluster)",
        barmode='overlay',
        yaxis=dict(showticklabels=False)
    )
    return fig

def get_segment_size_chart(df: pd.DataFrame) -> go.Figure:
    counts = df['segment_name'].value_counts().reset_index()
    counts.columns = ['segment_name', 'count']
    
    fig = px.pie(
        counts, names='segment_name', values='count', hole=0.5,
        color='segment_name', color_discrete_map=SEGMENT_COLORS,
        title="Segment Size Distribution",
        template='plotly_dark'
    )
    fig.update_traces(textposition='inside', textinfo='percent+label+value')
    return fig

def get_category_preference_chart(df: pd.DataFrame) -> go.Figure:
    counts = df.groupby(['segment_name', 'product_category_preference']).size().reset_index(name='count')
    totals = counts.groupby('segment_name')['count'].transform('sum')
    counts['proportion'] = counts['count'] / totals
    
    fig = px.bar(
        counts, x='segment_name', y='proportion', color='product_category_preference',
        barmode='stack', title="Product Category Preference by Segment",
        template='plotly_dark'
    )
    fig.update_layout(xaxis_title="Segment", yaxis_title="Proportion")
    return fig

def get_clv_ranking(df: pd.DataFrame) -> go.Figure:
    clv_means = df.groupby('segment_name')['clv_score'].mean().reset_index().sort_values('clv_score')
    
    fig = px.bar(
        clv_means, x='clv_score', y='segment_name', orientation='h',
        color='segment_name', color_discrete_map=SEGMENT_COLORS,
        text='clv_score', title="Customer Lifetime Value Score by Segment",
        template='plotly_dark'
    )
    fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    return fig

def get_kmeans_animation(df: pd.DataFrame) -> go.Figure:
    """Generates a custom animation of K-Means iterations on PCA data."""
    X_pca = df[['pca_x', 'pca_y']].values
    
    # Run custom k-means to capture history
    history = []
    
    # Init manually for first frame
    np.random.seed(42)
    initial_centers_idx = np.random.choice(len(X_pca), 5, replace=False)
    centers = X_pca[initial_centers_idx]
    
    from sklearn.metrics import pairwise_distances_argmin
    
    for i in range(1, 11):
        labels = pairwise_distances_argmin(X_pca, centers)
        
        # Save state
        frame_df = pd.DataFrame({'pca_x': X_pca[:, 0], 'pca_y': X_pca[:, 1], 'cluster': labels})
        frame_df['iteration'] = i
        history.append(frame_df)
        
        # Update centers
        new_centers = np.array([X_pca[labels == j].mean(axis=0) if sum(labels == j) > 0 else centers[j] for j in range(5)])
        centers = new_centers
        
    anim_df = pd.concat(history, ignore_index=True)
    anim_df['cluster'] = anim_df['cluster'].astype(str)
    
    fig = px.scatter(
        anim_df, x='pca_x', y='pca_y', color='cluster', 
        animation_frame='iteration', animation_group=anim_df.index % len(df),
        title="K-Means Convergence Animation (10 Iterations)",
        template='plotly_dark'
    )
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    return fig

def get_centroid_distances_chart(new_scaled_data, predicted_cluster) -> go.Figure:
    """Calculates distance from new customer to all centroids."""
    from sklearn.metrics import pairwise_distances
    distances = pairwise_distances(new_scaled_data, kmeans.cluster_centers_)[0]
    
    dist_df = pd.DataFrame({
        'segment_name': [SEGMENT_NAMES[i] for i in range(5)],
        'distance': distances,
        'is_predicted': [i == predicted_cluster for i in range(5)]
    }).sort_values('distance')
    
    fig = px.bar(
        dist_df, x='distance', y='segment_name', orientation='h',
        color='is_predicted', color_discrete_map={True: '#2ECC71', False: '#555555'},
        title="Distance to Segment Centroids (Lower is stronger match)",
        template='plotly_dark'
    )
    return fig

def get_dbscan_scatter(df: pd.DataFrame) -> (go.Figure, int):
    """Runs DBSCAN to find noise/outliers and returns plot + noise count."""
    X = df[CLUSTER_FEATURES].copy()
    X_scaled = scaler.transform(X)
    
    db = DBSCAN(eps=0.8, min_samples=10)
    labels = db.fit_predict(X_scaled)
    
    plot_df = df.copy()
    plot_df['is_noise'] = labels == -1
    noise_count = plot_df['is_noise'].sum()
    
    fig = px.scatter(
        plot_df, x='pca_x', y='pca_y', 
        color='is_noise', color_discrete_map={True: 'red', False: 'rgba(255,255,255,0.2)'},
        title="DBSCAN Outlier Detection (Red X = Noise)",
        template='plotly_dark'
    )
    
    # Update traces for noise points to be X
    fig.update_traces(selector=dict(marker_color='red'), marker=dict(symbol='x', size=8, opacity=1.0))
    fig.update_traces(selector=dict(marker_color='rgba(255,255,255,0.2)'), marker=dict(size=4))
    
    return fig, noise_count

def get_rfm_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates RFM quintiles."""
    res = df.copy()
    # 5 is best, 1 is worst. Recency: lower is better, so qcut labels are reversed.
    res['R'] = pd.qcut(res['recency_days'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop').astype(int)
    res['F'] = pd.qcut(res['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    res['M'] = pd.qcut(res['monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(int)
    res['rfm_score_str'] = res['R'].astype(str) + '-' + res['F'].astype(str) + '-' + res['M'].astype(str)
    res['rfm_sum'] = res['R'] + res['F'] + res['M']
    return res

def get_rfm_matrix(df: pd.DataFrame) -> go.Figure:
    """Creates a heatmap of Recency vs Frequency, colored by avg Monetary score."""
    df_rfm = get_rfm_scores(df)
    # Pivot to get R vs F and avg M
    matrix = df_rfm.groupby(['R', 'F'])['M'].mean().reset_index()
    pivot = matrix.pivot(index='R', columns='F', values='M')
    
    # Ensure full 5x5 grid
    for i in range(1, 6):
        if i not in pivot.index: pivot.loc[i] = np.nan
        if i not in pivot.columns: pivot[i] = np.nan
    
    pivot = pivot.sort_index(ascending=False) # R=5 at top
    pivot = pivot[sorted(pivot.columns)]      # F=1 to 5 left to right
    
    fig = px.imshow(
        pivot, 
        labels=dict(x="Frequency Score", y="Recency Score", color="Avg Monetary Score"),
        x=['1 (Low)', '2', '3', '4', '5 (High)'],
        y=['5 (Best)', '4', '3', '2', '1 (Worst)'],
        color_continuous_scale='RdYlGn',
        text_auto=".1f",
        title="RFM Matrix (Color = Avg Monetary Score)",
        template='plotly_dark'
    )
    return fig
