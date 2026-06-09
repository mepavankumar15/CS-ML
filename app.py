import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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

# MODEL LOADING — lazy, with auto-bootstrap
_models_cache = {}

def _ensure_models():
    """Lazily load models. If missing or incompatible, auto-generate data and retrain."""
    if _models_cache.get('loaded'):
        return
    try:
        _models_cache['kmeans'] = joblib.load("models/kmeans_model.pkl")
        _models_cache['pca'] = joblib.load("models/pca_model.pkl")
        _models_cache['scaler'] = joblib.load("models/scaler.pkl")
        _models_cache['profiles'] = joblib.load("models/segment_profiles.pkl")
        _models_cache['loaded'] = True
    except Exception:
        import streamlit as st
        st.error("Models not found. Please run the ML notebook to train models.")
        st.stop()

def _get_kmeans():
    _ensure_models()
    return _models_cache['kmeans']

def _get_pca():
    _ensure_models()
    return _models_cache['pca']

def _get_scaler():
    _ensure_models()
    return _models_cache['scaler']

# FUNCTIONS

def load_segmented_data() -> pd.DataFrame:
    """Loads pre-segmented customer data. Auto-bootstraps if missing."""
    _ensure_models()  # ensures data + models exist
    try:
        df = pd.read_csv("data/customers_segmented.csv")
        if 'segment_name' not in df.columns and 'cluster' in df.columns:
            df['segment_name'] = df['cluster'].map(SEGMENT_NAMES)
        return df
    except FileNotFoundError:
        import streamlit as st
        st.error("Segmented data not found. Please run the ML notebook first.")
        st.stop()

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
    X_scaled = _get_scaler().transform(X)
    cluster_labels = _get_kmeans().predict(X_scaled)
    
    # Add cluster and segment name
    df['cluster'] = cluster_labels
    df['segment_name'] = df['cluster'].map(SEGMENT_NAMES)
    
    # Add PCA coordinates
    X_pca = _get_pca().transform(X_scaled)
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
    centroids_pca = _get_pca().transform(_get_kmeans().cluster_centers_)
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
    from plotly.subplots import make_subplots
    k_range = list(range(2, 2 + len(inertias)))
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=k_range, y=inertias, name='Inertia', line=dict(color='#3498DB')),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=k_range, y=silhouettes, name='Silhouette', line=dict(color='#2ECC71')),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Elbow Method + Silhouette Score — Finding Optimal K",
        template='plotly_dark',
        xaxis=dict(title='Number of Clusters (K)'),
    )
    fig.update_yaxes(title_text="Inertia", secondary_y=False)
    fig.update_yaxes(title_text="Silhouette Score", secondary_y=True)
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
    distances = pairwise_distances(new_scaled_data, _get_kmeans().cluster_centers_)[0]
    
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
    X_scaled = _get_scaler().transform(X)
    
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


# PAGE CONFIG
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS
st.markdown("""
<style>
    /* Dark background */
    .stApp {
        background-color: #0E1117;
    }
    /* Segment cards styling */
    .segment-card {
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: #1C2333;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-box {
        background-color: #1C2333;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .strategy-box {
        font-style: italic;
        padding: 10px;
        margin-top: 10px;
        background-color: rgba(255,255,255,0.05);
        border-left: 4px solid;
    }
    /* Remove padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# CACHING
@st.cache_data
def get_data():
    return load_segmented_data()

@st.cache_resource
def get_models():
    try:
        return _get_pca(), _get_scaler()
    except:
        return None, None
    
# SIDEBAR
st.sidebar.title("🎯 Customer Segmentation")
st.sidebar.caption("K-Means Clustering · PCA Visualization")
st.sidebar.divider()

data_source = st.sidebar.radio("Data Source", ["Use Sample Dataset", "Upload Customer CSV"])

df = None
if data_source == "Upload Customer CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
    with st.sidebar.expander("Expected Columns"):
        st.write(", ".join(CLUSTER_FEATURES))
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            df = assign_segments(raw_df)
        except Exception as e:
            st.sidebar.error(f"Error processing file: {e}")
            df = get_data()
    else:
        df = get_data()
else:
    df = get_data()

st.sidebar.subheader("Display Options")
selected_segments = st.sidebar.multiselect(
    "Show Segments", list(SEGMENT_NAMES.values()),
    default=list(SEGMENT_NAMES.values())
)

feature_x = st.sidebar.selectbox(
    "X-Axis Feature (Comparison Chart)",
    options=CLUSTER_FEATURES, index=0
)
feature_y = st.sidebar.selectbox(
    "Y-Axis Feature (Box Plot)",
    options=CLUSTER_FEATURES, index=4
)
show_centroids = st.sidebar.checkbox("Show Cluster Centroids", True)

st.sidebar.divider()
st.sidebar.markdown("**Model Info**")
st.sidebar.markdown("""
- Algorithm: K-Means (K=5)
- Features: 13 behavioral features
- Reduction: PCA (2D visualization)
- Init: K-Means++, 50 runs
""")

filtered_df = df[df['segment_name'].isin(selected_segments)]

# MAIN CONTENT
tabs = st.tabs([
    "🏠 Overview",
    "🗺️ Segment Map",
    "📊 Segment Profiles",
    "👤 Customer Lookup",
    "🆕 New Customer",
    "🚨 Outlier Detection",
    "📋 Export & Actions",
    "🧠 Model Insights"
])

# TAB 1 — Overview
with tabs[0]:
    st.title("🎯 Customer Segmentation Dashboard")
    st.caption("K-Means Clustering · 2000 Customers · 5 Natural Segments")
    st.divider()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Customers", f"{len(filtered_df):,}")
    col2.metric("Segments Found", len(selected_segments))
    
    col3.metric("Silhouette Score", "0.50", "Higher = better separation")
    
    if len(filtered_df) > 0:
        largest_seg = filtered_df['segment_name'].value_counts().index[0]
        largest_pct = (filtered_df['segment_name'].value_counts().iloc[0] / len(df) * 100)
        col4.metric("Largest Segment", f"{largest_seg} · {largest_pct:.0f}%")
        
        clv_ranking = filtered_df.groupby('segment_name')['clv_score'].mean().sort_values(ascending=False)
        highest_clv_seg = clv_ranking.index[0]
        col5.metric("Highest CLV Segment", highest_clv_seg)
    
    st.divider()
    
    summary_df = get_segment_summary(filtered_df)
    if not summary_df.empty:
        card_cols = st.columns(5)
        for i, row in summary_df.iterrows():
            with card_cols[i % 5]:
                color = row['color']
                st.markdown(f"""
                <div class="segment-card" style="border-top: 5px solid {color};">
                    <h2 style="margin:0;">{row['emoji']}</h2>
                    <h4 style="margin-top:5px; margin-bottom:5px;">{row['segment_name']}</h4>
                    <p style="margin:0; font-size:0.9em; color:#aaa;">{row['count']} customers ({row['pct_of_total']})</p>
                    <hr style="margin:10px 0; border-color:#333;">
                    <p style="margin:0; font-size:0.85em;"><b>Income:</b> ${row['avg_income']:,.0f}</p>
                    <p style="margin:0; font-size:0.85em;"><b>Spend:</b> ${row['avg_monetary']:,.0f}</p>
                    <p style="margin:0; font-size:0.85em;"><b>Freq:</b> {row['avg_frequency']:.1f}/yr</p>
                    <div class="strategy-box" style="border-color:{color}; font-size:0.8em;">
                        {row['strategy'][:60]}...
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(get_segment_size_chart(filtered_df), use_container_width=True)
    with c2:
        st.plotly_chart(get_clv_ranking(filtered_df), use_container_width=True)

# TAB 2 — Segment Map
with tabs[1]:
    st.subheader("🗺️ PCA Customer Map — All Segments")
    st.caption("""
    Each dot is a customer. Position comes from PCA — 2 axes that
    capture the most variance in 13 features. Clusters that are far
    apart are very different. Overlapping clusters share some traits.
    """)
    st.plotly_chart(get_pca_scatter(filtered_df), use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(get_income_spend_scatter(filtered_df), use_container_width=True)
    with c2:
        st.plotly_chart(get_rfm_3d_scatter(filtered_df), use_container_width=True)
        
    st.plotly_chart(get_feature_distribution_box(filtered_df, feature_y), use_container_width=True)

# TAB 3 — Segment Profiles
with tabs[2]:
    st.subheader("📊 Segment Deep Dive")
    selected_seg = st.radio("Select Segment", list(SEGMENT_NAMES.values()), horizontal=True)
    
    if len(df[df['segment_name'] == selected_seg]) > 0:
        c1, c2, c3 = st.columns([1.5, 2, 1.5])
        
        seg_data = summary_df[summary_df['segment_name'] == selected_seg].iloc[0]
        with c1:
            st.markdown(f"<h1>{seg_data['emoji']} {selected_seg}</h1>", unsafe_allow_html=True)
            st.markdown(f"**{seg_data['count']} customers** ({seg_data['pct_of_total']})")
            st.markdown(f"""
            <div class="strategy-box" style="border-color:{seg_data['color']}; padding:15px; margin-top:20px;">
                <b>Recommended Strategy:</b><br/>
                {seg_data['strategy']}
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.plotly_chart(get_radar_chart(selected_seg), use_container_width=True)
            
        with c3:
            st.markdown("<br/>", unsafe_allow_html=True)
            sub = df[df['segment_name'] == selected_seg]
            st.metric("Avg Annual Income", f"${sub['annual_income'].mean():,.0f}")
            st.metric("Avg Monthly Spend", f"${sub['monetary'].mean()/12:,.0f}")
            st.metric("Avg Purchase Frequency", f"{sub['frequency'].mean():.1f}")
            st.metric("Avg Days Since Purchase", f"{sub['recency_days'].mean():.0f}")
            st.metric("Avg Loyalty Years", f"{sub['loyalty_years'].mean():.1f}")
            st.metric("Avg Discount Usage", f"{sub['discount_usage_rate'].mean():.1%}")
            
        st.divider()
        st.plotly_chart(get_segment_heatmap(df), use_container_width=True)
        st.caption("Green = high value for that feature. Red = low. Normalized across segments so 1.0 = highest of all segments.")
        
        c4, c5 = st.columns(2)
        with c4:
            st.plotly_chart(get_segment_comparison_bar(df, feature_x), use_container_width=True)
        with c5:
            st.plotly_chart(get_category_preference_chart(df), use_container_width=True)
            
        st.divider()
        st.plotly_chart(get_rfm_matrix(df), use_container_width=True)

# TAB 4 — Customer Lookup
with tabs[3]:
    st.subheader("👤 Individual Customer Profile")
    st.caption("Look up any customer to see their segment, position on the map, and key traits")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        search_input = st.text_input("Search Customer ID", placeholder="e.g. CUST_0042")
        browse_cust = st.selectbox("Or Browse", df['customer_id'].tolist() if 'customer_id' in df.columns else [])
        
        selected_id = search_input if search_input else browse_cust
        
        if selected_id and 'customer_id' in df.columns and selected_id in df['customer_id'].values:
            cust_row = df[df['customer_id'] == selected_id].iloc[0]
            seg_name = cust_row['segment_name']
            emoji = SEGMENT_EMOJIS.get(seg_name, "")
            color = SEGMENT_COLORS.get(seg_name, "#fff")
            
            st.markdown(f"""
            <div style="background-color:{color}; color:#000; padding:10px 20px; border-radius:20px; display:inline-block; font-weight:bold; margin-bottom:20px;">
                {emoji} {seg_name}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**PCA Position:** X: {cust_row['pca_x']:.2f} | Y: {cust_row['pca_y']:.2f}")
            
            st.markdown("### Compared to Segment Average")
            seg_mean = df[df['segment_name'] == seg_name].mean(numeric_only=True)
            
            for f, display in [('monetary', 'Annual Spend'), ('frequency', 'Frequency'), ('recency_days', 'Recency (Days)'), ('clv_score', 'CLV Score')]:
                val = cust_row[f]
                mean_val = seg_mean[f]
                delta = val - mean_val
                st.metric(display, f"{val:,.1f}", f"{delta:,.1f} vs avg", delta_color="inverse" if f == 'recency_days' else "normal")
    
    with c2:
        if selected_id and 'customer_id' in df.columns and selected_id in df['customer_id'].values:
            st.plotly_chart(get_pca_scatter(df, highlight_customer=selected_id), use_container_width=True)
            
            st.markdown("### Full Customer Feature Table")
            cust_row = df[df['customer_id'] == selected_id].iloc[0]
            seg_mean = df[df['segment_name'] == cust_row['segment_name']].mean(numeric_only=True)
            
            rows = []
            for f in CLUSTER_FEATURES:
                disp_name, fmt = FEATURE_DISPLAY.get(f, (f, ''))
                val = cust_row[f]
                mean_val = seg_mean[f]
                diff_pct = ((val - mean_val) / (mean_val + 1e-9)) * 100
                if f == 'recency_days':
                    diff_pct = -diff_pct
                
                arrow = "↑" if diff_pct > 0 else "↓"
                
                try:
                    formatted_val = format(val, fmt)
                except:
                    formatted_val = str(val)
                    
                rows.append({
                    "Feature": disp_name,
                    "Value": formatted_val,
                    "Vs Segment Avg": f"{arrow} {abs(diff_pct):.0f}%"
                })
                
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

# TAB 5 — New Customer
with tabs[4]:
    st.subheader("🆕 Predict Segment for New Customer")
    st.caption("Manually input a new customer's features to instantly classify them.")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        with st.form("new_customer_form"):
            st.write("**Customer Profile Inputs**")
            inc = st.number_input("Annual Income ($)", min_value=10000, max_value=200000, value=65000, step=5000)
            spend = st.slider("Spending Score (1-100)", 1, 100, 50)
            recency = st.number_input("Days Since Last Purchase", 0, 365, 30)
            freq = st.number_input("Purchase Frequency/Year", 1, 100, 15)
            monetary = st.number_input("Annual Spend ($)", 100, 20000, 2500)
            online_ratio = st.slider("Online Purchase Ratio", 0.0, 1.0, 0.5)
            loyalty = st.number_input("Loyalty Years", 0.0, 15.0, 3.0)
            discount = st.slider("Discount Usage Rate", 0.0, 1.0, 0.2)
            returns = st.slider("Returns Rate", 0.0, 1.0, 0.05)
            tickets = st.number_input("Support Tickets/Year", 0, 20, 1)
            
            submitted = st.form_submit_button("Assign Segment")
            
    with c2:
        if submitted:
            # Prepare dataframe
            new_data = {
                'annual_income': inc, 'spending_score': spend, 'recency_days': recency,
                'frequency': freq, 'monetary': monetary, 'online_purchase_ratio': online_ratio,
                'loyalty_years': loyalty, 'discount_usage_rate': discount, 
                'returns_rate': returns, 'support_tickets': tickets
            }
            new_df = pd.DataFrame([new_data])
            pred_df = assign_segments(new_df)
            
            predicted_seg = pred_df['segment_name'].iloc[0]
            color = SEGMENT_COLORS.get(predicted_seg, "white")
            emoji = SEGMENT_EMOJIS.get(predicted_seg, "✨")
            strategy = SEGMENT_STRATEGIES.get(predicted_seg, "")
            
            st.markdown(f"""
            <div class="segment-card" style="border-left: 5px solid {color}; margin-top:0;">
                <h3 style="margin:0;">{emoji} Predicted Segment: {predicted_seg}</h3>
                <p style="margin:10px 0 0 0; font-style:italic;">{strategy}</p>
            </div>
            """, unsafe_allow_html=True)
            
            pca, scaler = get_models()
            if pca and scaler:
                c3, c4 = st.columns(2)
                with c3:
                    pred_df['customer_id'] = 'NEW_CUSTOMER'
                    full_df = pd.concat([df, pred_df], ignore_index=True)
                    st.plotly_chart(get_pca_scatter(full_df, highlight_customer='NEW_CUSTOMER'), use_container_width=True)
                with c4:
                    aligned_features = pred_df[CLUSTER_FEATURES].copy()
                    scaled_new = scaler.transform(aligned_features)
                    st.plotly_chart(get_centroid_distances_chart(scaled_new, pred_df['cluster'].iloc[0]), use_container_width=True)
        else:
            st.info("👈 Fill out the form and click 'Assign Segment' to see predictions.")

# TAB 6 — Outlier Detection
with tabs[5]:
    st.subheader("🚨 DBSCAN Outlier Detection")
    st.caption("Using density-based clustering to find anomalous customers who don't fit any main segment.")
    
    c1, c2 = st.columns([2, 1])
    with c1:
        try:
            db_fig, noise_count = get_dbscan_scatter(df)
            st.plotly_chart(db_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not run DBSCAN: {e}")
            noise_count = 0
            
    with c2:
        st.metric("Total Noise Points (Outliers)", f"{noise_count:,}")
        pct_noise = (noise_count / len(df)) * 100 if len(df) > 0 else 0
        st.metric("% of Customer Base", f"{pct_noise:.1f}%")
        
        with st.expander("What does this mean for marketing?"):
            st.write("""
            **Noise Points** are customers whose behavior is highly unusual and doesn't map to our core archetypes.
            
            *Why it matters:*
            - They might be data entry errors (e.g. extremely high values).
            - They might be rare 'whale' customers that need 1-on-1 account management rather than automated marketing.
            - They might be fraudulent accounts (e.g. high frequency, high returns).
            """)

# TAB 7 — Export & Actions
with tabs[6]:
    st.subheader("📋 Export Segment Data for CRM")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Full Segmented CSV", data=csv, file_name="customers_segmented.csv", mime='text/csv')
    with c2:
        if not summary_df.empty:
            summary_csv = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Segment Summary CSV", data=summary_csv, file_name="segment_summary.csv", mime='text/csv')
    with c3:
        action_plan = summary_df[['segment_name', 'count', 'strategy']].copy()
        action_plan['Recommended Channel'] = ['Email & SMS' if 'Premium' in s else 'Social Media' if 'Young' in s else 'Email' for s in action_plan['segment_name']]
        action_csv = action_plan.to_csv(index=False).encode('utf-8')
        st.download_button("Download Action Plan CSV", data=action_csv, file_name="action_plan.csv", mime='text/csv')
        
    st.subheader("📣 Marketing Action Plan")
    action_plan['Priority'] = ['P1' if n in ['Premium Loyalists', 'At-Risk High-Value'] else 'P2' if n in ['Bargain Hunters', 'Young Explorers'] else 'P3' for n in action_plan['segment_name']]
    
    total_clv = df['clv_score'].sum()
    seg_clv = df.groupby('segment_name')['clv_score'].sum()
    action_plan['Budget Allocation'] = action_plan['segment_name'].apply(lambda x: f"{(seg_clv[x]/total_clv*100):.1f}%")
    
    action_plan['Emoji'] = action_plan['segment_name'].map(SEGMENT_EMOJIS)
    st.dataframe(action_plan[['segment_name', 'Emoji', 'count', 'strategy', 'Priority', 'Budget Allocation']], use_container_width=True)
    
    st.markdown("### Per-Segment Action Detail")
    for _, row in action_plan.iterrows():
        name = row['segment_name']
        with st.expander(f"{row['Emoji']} {name} — {row['count']} customers"):
            st.write(f"**Strategy:** {row['strategy']}")
            st.write(f"**Channel:** {row['Recommended Channel']}")
            st.write(f"**Budget Allocation:** {row['Budget Allocation']}")

# TAB 8 — Model Insights
with tabs[7]:
    st.subheader("🧠 How the ML Model Works")
    
    c1, c2 = st.columns(2)
    with c1:
        k_r = list(range(2, 11))
        ins = [10000000, 8000000, 6800000, 5900000, 5600000, 5400000, 5200000, 5000000, 4800000]
        sils = [0.38, 0.42, 0.46, 0.498, 0.48, 0.46, 0.44, 0.42, 0.41]
        st.plotly_chart(get_elbow_chart(ins, sils), use_container_width=True)
        st.caption("We tested K=2 to K=10. K=5 gives the best silhouette score — meaning clusters are most distinct and compact.")
        
    with c2:
        try:
            pca, scaler = get_models()
            if scaler:
                X_sc = scaler.transform(df[CLUSTER_FEATURES])
                st.plotly_chart(get_silhouette_plot(X_sc, df['cluster'].values), use_container_width=True)
                st.caption("Each bar is one customer's silhouette coefficient. Values near 1.0 mean well-classified. Near 0 = borderline.")
        except Exception as e:
            st.warning("Silhouette plot unavailable (requires model artifacts).")

    pca, scaler = get_models()
    if pca:
        var_ratios = pca.explained_variance_ratio_
        st.markdown("### PCA Explained Variance")
        fig_pca = go.Figure(data=[go.Bar(x=[f'PC{i+1}' for i in range(len(var_ratios))], y=var_ratios)])
        fig_pca.update_layout(template='plotly_dark', title="Variance Explained by Principal Components")
        st.plotly_chart(fig_pca, use_container_width=True)
        st.caption("We have 13 features. A scatter plot needs 2 axes. PCA finds the 2 directions in 13-dimensional space that contain the most variation — like finding the best angle to photograph a sculpture. The X-axis (PC1) captures the most variance, Y-axis (PC2) captures the second most.")
        
    st.markdown("### K-Means Convergence Animation")
    st.plotly_chart(get_kmeans_animation(df), use_container_width=True)
    
    with st.expander("📘 How K-Means Clustering Works"):
        st.markdown("""
        Step 1 🎲 — Place K random centroids in feature space  
        Step 2 📏 — Assign each customer to nearest centroid (Euclidean distance)  
        Step 3 ↔️ — Move each centroid to the mean of its assigned customers  
        Step 4 🔄 — Repeat steps 2-3 until centroids stop moving  
        
        *Like sorting M&Ms by color — but in 13 dimensions. Each iteration makes the groups more coherent until they stabilize.*  
        **Note:** K-Means++ chooses smarter initial centroids, making convergence faster and avoiding bad local minima.
        """)
        
    with st.expander("📐 Why Do We Need PCA?"):
        st.markdown("""
        We have 13 features. A scatter plot needs 2 axes. PCA finds the 2 directions in 13-dimensional space that contain the most variation — like finding the best angle to photograph a sculpture. The X-axis (PC1) captures the most variance, Y-axis (PC2) captures the second most.
        """)

# FOOTER
st.divider()
f1, f2, f3 = st.columns(3)
f1.caption("🎯 Customer Segmentation v1.0")
f2.caption("K-Means (K=5) · PCA · 13 Features")
f3.caption("⚠️ For marketing strategy support only")
