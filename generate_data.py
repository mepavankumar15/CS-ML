"""
generate_data.py
Generates a synthetic customer dataset for segmentation analysis.
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)
N_CUSTOMERS = 2000

def generate_archetype(n: int, name: str, params: dict) -> pd.DataFrame:
    """
    Generates a DataFrame for a specific customer archetype based on provided parameters.
    
    Args:
        n (int): Number of customers to generate.
        name (str): Label for the archetype (e.g., 'A').
        params (dict): Dictionary defining bounds/means for features.
        
    Returns:
        pd.DataFrame: Generated customer data.
    """
    df = pd.DataFrame()
    df['archetype'] = [name] * n
    
    # Base features (same for all unless specified)
    df['age'] = np.clip(np.random.normal(42, 13, n), 18, 75).astype(int)
    df['gender'] = np.random.choice(['F', 'M'], n, p=[0.55, 0.45])
    
    categories = ['Electronics', 'Clothing', 'Grocery', 'Home', 'Beauty', 'Sports', 'Books']
    df['product_category_preference'] = np.random.choice(categories, n)
    df['support_tickets'] = np.clip(np.random.normal(2, 2, n), 0, 20).astype(int)
    
    # Generate continuous features with noise
    continuous_features = [
        'annual_income', 'spending_score', 'recency_days', 'frequency', 
        'monetary', 'discount_usage_rate', 'loyalty_years', 'returns_rate', 
        'online_purchase_ratio'
    ]
    
    for feat in continuous_features:
        if feat in params:
            low, high = params[feat]
            # using uniform distribution as base
            base_values = np.random.uniform(low, high, n)
            # Add Gaussian noise (5% of range)
            noise_std = (high - low) * 0.05
            noisy_values = base_values + np.random.normal(0, noise_std, n)
            
            # Clip back to realistic absolute ranges to ensure valid values
            clip_low = max(0, low - noise_std * 2) 
            clip_high = high + noise_std * 2
            
            # Specific absolute bounds to enforce sanity
            if feat == 'spending_score': clip_low, clip_high = 1, 100
            if feat == 'recency_days': clip_low, clip_high = 1, 365
            if feat == 'discount_usage_rate': clip_low, clip_high = 0.0, 1.0
            if feat == 'returns_rate': clip_low, clip_high = 0.0, 1.0
            if feat == 'online_purchase_ratio': clip_low, clip_high = 0.0, 1.0
            if feat == 'frequency': clip_low = 1
            if feat == 'monetary': clip_low = 50
            if feat == 'loyalty_years': clip_low, clip_high = 0, 15
            
            df[feat] = np.clip(noisy_values, clip_low, clip_high)
        else:
            # Default fallback if not defined in archetype params
            if feat == 'online_purchase_ratio':
                df[feat] = np.clip(np.random.uniform(0.1, 0.9, n) + np.random.normal(0, 0.05, n), 0, 1)
            else:
                df[feat] = 0
                
    # Formatting / Rounding where appropriate
    df['annual_income'] = df['annual_income'].round(2)
    df['spending_score'] = df['spending_score'].astype(int)
    df['recency_days'] = df['recency_days'].astype(int)
    df['frequency'] = df['frequency'].astype(int)
    df['monetary'] = df['monetary'].round(2)
    df['loyalty_years'] = df['loyalty_years'].astype(int)
    
    return df

def generate_data() -> None:
    """
    Generates the entire dataset by combining archetypes and deriving new features.
    """
    print("Generating archetypes...")
    
    # Archetype Definitions
    archetypes_def = {
        'A': { # Premium Loyalists
            'n': 400,
            'annual_income': (80000, 200000),
            'spending_score': (75, 100),
            'recency_days': (1, 30),
            'frequency': (30, 80),
            'monetary': (3000, 15000),
            'discount_usage_rate': (0.0, 0.1),
            'loyalty_years': (5, 15),
            'returns_rate': (0.0, 0.05),
            'online_purchase_ratio': (0.4, 0.8) # Sensible default
        },
        'B': { # Occasional Shoppers
            'n': 500,
            'annual_income': (30000, 60000),
            'spending_score': (35, 65),
            'recency_days': (60, 180),
            'frequency': (3, 15),
            'monetary': (200, 1200),
            'discount_usage_rate': (0.2, 0.5),
            'loyalty_years': (0, 3),
            'returns_rate': (0.05, 0.15),
            'online_purchase_ratio': (0.2, 0.6)
        },
        'C': { # Bargain Hunters
            'n': 450,
            'annual_income': (20000, 50000),
            'spending_score': (40, 70),
            'recency_days': (1, 45),
            'frequency': (20, 60),
            'monetary': (500, 3000),
            'discount_usage_rate': (0.6, 1.0),
            'loyalty_years': (2, 8),
            'returns_rate': (0.1, 0.3),
            'online_purchase_ratio': (0.4, 0.9)
        },
        'D': { # At-Risk High-Value
            'n': 350,
            'annual_income': (70000, 150000),
            'spending_score': (10, 35),
            'recency_days': (180, 365),
            'frequency': (1, 6),
            'monetary': (1500, 8000),
            'discount_usage_rate': (0.0, 0.2),
            'loyalty_years': (3, 12),
            'returns_rate': (0.15, 0.4),
            'online_purchase_ratio': (0.3, 0.7)
        },
        'E': { # Young Explorers
            'n': 300,
            'annual_income': (15000, 35000),
            'spending_score': (60, 95),
            'recency_days': (1, 60),
            'frequency': (8, 25),
            'monetary': (100, 800),
            'discount_usage_rate': (0.3, 0.8),
            'loyalty_years': (0, 2),
            'returns_rate': (0.0, 0.1),
            'online_purchase_ratio': (0.7, 1.0)
        }
    }
    
    dfs = []
    archetype_counts = {}
    
    for name, a_def in archetypes_def.items():
        n = a_def['n']
        archetype_counts[name] = n
        params = {k: v for k, v in a_def.items() if k != 'n'}
        dfs.append(generate_archetype(n, name, params))
        
    df = pd.concat(dfs, ignore_index=True)
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Assign customer_id formatted as CUST_0001
    df.insert(0, 'customer_id', [f"CUST_{i+1:04d}" for i in range(len(df))])
    
    # STEP 4: Add derived features
    # avg_order_value: monetary / frequency (clip to min 1.0)
    df['avg_order_value'] = np.clip(df['monetary'] / df['frequency'], 1.0, None).round(2)
    
    # clv_score: (frequency * monetary) / (recency_days + 1)
    df['clv_score'] = ((df['frequency'] * df['monetary']) / (df['recency_days'] + 1)).round(2)
    
    # engagement_score: (1/recency_days * 100) + (frequency * 2) + (online_purchase_ratio * 20) - (support_tickets * 3)
    eng_score = (1 / df['recency_days'] * 100) + (df['frequency'] * 2) + \
                (df['online_purchase_ratio'] * 20) - (df['support_tickets'] * 3)
    # clip to 0-200
    df['engagement_score'] = np.clip(eng_score, 0, 200).round(2)
    
    # STEP 5: Shuffle and save
    os.makedirs("data", exist_ok=True)
    output_path = "data/customers.csv"
    df.to_csv(output_path, index=False)
    
    print("Customer dataset generated:")
    print(f"  Total customers: {N_CUSTOMERS}")
    print(f"  Archetypes: A={archetype_counts['A']}, B={archetype_counts['B']}, C={archetype_counts['C']}, D={archetype_counts['D']}, E={archetype_counts['E']}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Avg annual income: ${df['annual_income'].mean():,.0f}")
    print(f"  Avg monetary: ${df['monetary'].mean():,.0f}")
    print(f"Data saved to {output_path} [OK]")

if __name__ == "__main__":
    try:
        generate_data()
    except Exception as e:
        print(f"Error generating data: {e}")
