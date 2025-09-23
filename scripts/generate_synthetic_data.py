"""
Synthetic Data Generation Script for EABL Insights Customer Segmentation Challenge
Generates realistic customer data with Kenyan context and 2025-specific features
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import os

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
fake = Faker()
Faker.seed(42)

# Kenyan-specific data
KENYAN_COUNTIES = [
    'Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret', 'Thika', 'Malindi',
    'Kitale', 'Garissa', 'Kakamega', 'Machakos', 'Meru', 'Nyeri', 'Kericho',
    'Embu', 'Migori', 'Homa Bay', 'Bungoma', 'Kilifi', 'Kwale'
]

INCOME_BRACKETS = [
    "Below 20k", "20k-50k", "50k-100k", "100k-200k", "200k-500k", "Above 500k"
]

BEHAVIOR_CATEGORIES = [
    "Tusker,Vooma", "Tusker,Guinness", "White Cap,Pilsner", "Premium Spirits",
    "Local Brews", "Mixed Portfolio", "Occasional Drinker", "Social Drinker",
    "Premium Consumer", "Budget Conscious"
]

PROFIT_SEGMENTS = [
    "High-Spirits", "Premium-Beer", "Volume-Driver", "Emerging-Consumer",
    "Loyal-Traditional", "Price-Sensitive", "Occasional-Premium"
]

SEGMENT_TARGETS = [
    "Urban Youth", "Rural Traditional", "Premium Urban", "Budget Conscious",
    "Social Millennials", "Corporate Professional", "Weekend Warriors",
    "Loyal Veterans", "Price Seekers", "Premium Explorers"
]

def generate_customer_data(n_samples=80000):
    """Generate synthetic customer data with realistic distributions"""
    
    print(f"Generating {n_samples:,} synthetic customer records...")
    
    data = []
    
    for i in range(n_samples):
        # Basic demographics with realistic correlations
        age = np.random.normal(35, 12)
        age = max(18, min(70, int(age)))  # Clamp between 18-70
        
        # Income correlation with age (peak earning years)
        if age < 25:
            income_weights = [0.4, 0.4, 0.15, 0.04, 0.01, 0.0]
        elif age < 35:
            income_weights = [0.2, 0.35, 0.25, 0.15, 0.04, 0.01]
        elif age < 50:
            income_weights = [0.1, 0.25, 0.3, 0.25, 0.08, 0.02]
        else:
            income_weights = [0.15, 0.3, 0.25, 0.2, 0.08, 0.02]
        
        income_bracket = np.random.choice(INCOME_BRACKETS, p=income_weights)
        
        # Purchase history correlation with income
        income_multiplier = {
            "Below 20k": 0.5, "20k-50k": 0.8, "50k-100k": 1.2,
            "100k-200k": 1.8, "200k-500k": 2.5, "Above 500k": 3.5
        }
        
        base_purchases = np.random.poisson(8)
        purchase_history = max(1, int(base_purchases * income_multiplier[income_bracket]))
        
        # Behavior categories with income correlation
        if "Above 500k" in income_bracket or "200k-500k" in income_bracket:
            behavior = np.random.choice([
                "Premium Spirits", "Premium Consumer", "Tusker,Guinness", "Mixed Portfolio"
            ])
        elif "Below 20k" in income_bracket:
            behavior = np.random.choice([
                "Local Brews", "Budget Conscious", "Price-Sensitive", "Occasional Drinker"
            ])
        else:
            behavior = np.random.choice(BEHAVIOR_CATEGORIES)
        
        # Engagement rate with some correlation to behavior and age
        base_engagement = 0.3
        if "Premium" in behavior:
            base_engagement += 0.2
        if age < 35:
            base_engagement += 0.15
        if "Urban" in behavior or "Social" in behavior:
            base_engagement += 0.1
            
        engagement_rate = min(0.95, max(0.05, np.random.normal(base_engagement, 0.15)))
        
        # County with urban bias for higher income
        if income_bracket in ["200k-500k", "Above 500k"]:
            county = np.random.choice(['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru'], 
                                    p=[0.6, 0.2, 0.1, 0.1])
        else:
            county = np.random.choice(KENYAN_COUNTIES)
        
        # 2025-specific fields
        # Profit segment correlation with behavior and income
        if "Premium" in behavior and income_bracket in ["200k-500k", "Above 500k"]:
            profit_segment = np.random.choice(["High-Spirits", "Premium-Beer"], p=[0.7, 0.3])
        elif "Budget" in behavior or "Below 20k" in income_bracket:
            profit_segment = np.random.choice(["Price-Sensitive", "Volume-Driver"], p=[0.6, 0.4])
        else:
            profit_segment = np.random.choice(PROFIT_SEGMENTS)
        
        # M-Pesa upgrade engagement boost (2025 feature)
        # Higher for urban, younger, tech-savvy segments
        base_upgrade = 1.0
        if county in ['Nairobi', 'Mombasa', 'Kisumu']:
            base_upgrade += 0.3
        if age < 40:
            base_upgrade += 0.2
        if engagement_rate > 0.6:
            base_upgrade += 0.15
            
        upgrade_engagement = max(0.8, min(2.0, np.random.normal(base_upgrade, 0.2)))
        
        # Segment target based on multiple factors
        if age < 30 and county in ['Nairobi', 'Mombasa'] and engagement_rate > 0.5:
            segment_target = "Urban Youth"
        elif income_bracket in ["200k-500k", "Above 500k"] and "Premium" in behavior:
            segment_target = "Premium Urban"
        elif age > 45 and purchase_history > 20:
            segment_target = "Loyal Veterans"
        elif "Budget" in behavior or "Price" in behavior:
            segment_target = "Budget Conscious"
        elif age < 35 and engagement_rate > 0.6:
            segment_target = "Social Millennials"
        else:
            segment_target = np.random.choice(SEGMENT_TARGETS)
        
        # Create customer record
        customer = {
            'customer_id': 1000 + i,
            'age': age,
            'income_kes': income_bracket,
            'purchase_history': purchase_history,
            'behavior': behavior,
            'engagement_rate': round(engagement_rate, 3),
            'county': county,
            'profit_segment': profit_segment,
            'upgrade_engagement': round(upgrade_engagement, 3),
            'segment_target': segment_target
        }
        
        data.append(customer)
        
        if (i + 1) % 10000 == 0:
            print(f"Generated {i + 1:,} records...")
    
    df = pd.DataFrame(data)
    print(f"\nDataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df

def create_train_test_split(df, test_size=0.2):
    """Create train/test split with stratification on segment_target"""
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['segment_target'],
        random_state=42
    )
    
    print(f"Train set: {train_df.shape[0]:,} records")
    print(f"Test set: {test_df.shape[0]:,} records")
    
    return train_df, test_df

def save_datasets(train_df, test_df, data_dir):
    """Save train and test datasets to CSV files"""
    os.makedirs(data_dir, exist_ok=True)
    
    # Save training data (with target labels)
    train_path = os.path.join(data_dir, 'train_data.csv')
    train_df.to_csv(train_path, index=False)
    print(f"Training data saved to: {train_path}")
    
    # Save test data (without target labels for challenge simulation)
    test_features = test_df.drop('segment_target', axis=1)
    test_path = os.path.join(data_dir, 'test_data.csv')
    test_features.to_csv(test_path, index=False)
    print(f"Test data saved to: {test_path}")
    
    # Save test labels separately (for evaluation)
    test_labels_path = os.path.join(data_dir, 'test_labels.csv')
    test_df[['customer_id', 'segment_target']].to_csv(test_labels_path, index=False)
    print(f"Test labels saved to: {test_labels_path}")
    
    # Save sample submission format
    sample_submission = test_df[['customer_id']].copy()
    sample_submission['segment_target'] = 'Urban Youth'  # Placeholder
    sample_path = os.path.join(data_dir, 'sample_submission.csv')
    sample_submission.to_csv(sample_path, index=False)
    print(f"Sample submission saved to: {sample_path}")

def generate_data_summary(df):
    """Generate and display data summary statistics"""
    print("\n" + "="*60)
    print("TUSKER LOYALTY DATASET SUMMARY")
    print("="*60)
    
    print(f"\nDataset Overview:")
    print(f"- Total Records: {len(df):,}")
    print(f"- Features: {len(df.columns)}")
    print(f"- Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\nAge Distribution:")
    print(f"- Mean: {df['age'].mean():.1f} years")
    print(f"- Std: {df['age'].std():.1f} years")
    print(f"- Range: {df['age'].min()}-{df['age'].max()} years")
    
    print(f"\nIncome Distribution:")
    income_dist = df['income_kes'].value_counts(normalize=True)
    for income, pct in income_dist.items():
        print(f"- {income}: {pct:.1%}")
    
    print(f"\nTop Counties:")
    county_dist = df['county'].value_counts().head()
    for county, count in county_dist.items():
        print(f"- {county}: {count:,} ({count/len(df):.1%})")
    
    print(f"\nSegment Target Distribution:")
    segment_dist = df['segment_target'].value_counts()
    for segment, count in segment_dist.items():
        print(f"- {segment}: {count:,} ({count/len(df):.1%})")
    
    print(f"\nEngagement Statistics:")
    print(f"- Mean Engagement Rate: {df['engagement_rate'].mean():.3f}")
    print(f"- Mean Upgrade Engagement: {df['upgrade_engagement'].mean():.3f}")
    print(f"- Mean Purchase History: {df['purchase_history'].mean():.1f}")

if __name__ == "__main__":
    print("🍺 TUSKER LOYALTY DATA GENERATION")
    print("Generating synthetic customer data for segmentation challenge...")
    print("-" * 60)
    
    # Generate the dataset
    df = generate_customer_data(n_samples=80000)
    
    # Create train/test split
    train_df, test_df = create_train_test_split(df)
    
    # Save datasets
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    save_datasets(train_df, test_df, data_dir)
    
    # Generate summary
    generate_data_summary(df)
    
    print("\n✅ Data generation completed successfully!")
    print(f"📁 Files saved to: {data_dir}")
    print("\nNext steps:")
    print("1. Run the EDA notebook to explore the data")
    print("2. Train clustering models to achieve Silhouette score >0.85")
    print("3. Test the web application with generated data")
