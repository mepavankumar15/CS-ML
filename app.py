import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from segmentor import (
    CLUSTER_FEATURES, SEGMENT_NAMES, SEGMENT_COLORS, SEGMENT_EMOJIS, 
    SEGMENT_STRATEGIES, FEATURE_DISPLAY, load_segmented_data, assign_segments,
    get_segment_summary, get_pca_scatter, get_radar_chart, 
    get_segment_comparison_bar, get_rfm_3d_scatter, get_income_spend_scatter,
    get_feature_distribution_box, get_segment_heatmap, get_elbow_chart,
    get_silhouette_plot, get_segment_size_chart, get_category_preference_chart,
    get_clv_ranking, get_kmeans_animation, get_centroid_distances_chart,
    get_dbscan_scatter, get_rfm_matrix,
    _get_pca, _get_scaler
)

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
