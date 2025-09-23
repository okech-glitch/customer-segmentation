"""
Evaluation Script for EABL Insights Customer Segmentation Challenge
Computes Silhouette scores and creates leaderboard simulation (Target: >0.6)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
import json
import os
from datetime import datetime

class SegmentationEvaluator:
    """Evaluates customer segmentation submissions"""
    
    def __init__(self, test_labels_path="../data/test_labels.csv"):
        """Initialize evaluator with test labels"""
        self.test_labels_path = test_labels_path
        self.load_test_labels()
        
    def load_test_labels(self):
        """Load ground truth test labels"""
        try:
            self.test_labels = pd.read_csv(self.test_labels_path)
            print(f"✅ Loaded {len(self.test_labels)} test labels")
        except FileNotFoundError:
            print(f"⚠️ Test labels file not found: {self.test_labels_path}")
            self.test_labels = None
    
    def evaluate_submission(self, submission_path, features_path="../data/test_data.csv"):
        """Evaluate a submission file"""
        
        try:
            # Load submission and features
            submission = pd.read_csv(submission_path)
            features = pd.read_csv(features_path)
            
            # Validate submission format
            if not self._validate_submission(submission):
                return None
            
            # Merge with features for clustering evaluation
            merged_data = features.merge(submission, on='customer_id', how='inner')
            
            # Preprocess features for silhouette calculation
            X_processed = self._preprocess_features(merged_data)
            
            # Calculate metrics
            metrics = self._calculate_metrics(merged_data, X_processed)
            
            return metrics
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return None
    
    def _validate_submission(self, submission):
        """Validate submission file format"""
        
        required_cols = ['customer_id', 'segment_target']
        missing_cols = [col for col in required_cols if col not in submission.columns]
        
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return False
        
        if submission['customer_id'].duplicated().any():
            print("❌ Duplicate customer IDs found")
            return False
        
        if submission['segment_target'].isnull().any():
            print("❌ Missing segment predictions found")
            return False
        
        print("✅ Submission format valid")
        return True
    
    def _preprocess_features(self, df):
        """Preprocess features for clustering evaluation"""
        
        # Encode categorical variables
        le_income = LabelEncoder()
        le_behavior = LabelEncoder()
        le_county = LabelEncoder()
        le_profit = LabelEncoder()
        
        df_processed = df.copy()
        df_processed['income_encoded'] = le_income.fit_transform(df_processed['income_kes'])
        df_processed['behavior_encoded'] = le_behavior.fit_transform(df_processed['behavior'])
        df_processed['county_encoded'] = le_county.fit_transform(df_processed['county'])
        df_processed['profit_encoded'] = le_profit.fit_transform(df_processed['profit_segment'])
        
        # Select numerical features
        feature_cols = [
            'age', 'income_encoded', 'purchase_history', 'behavior_encoded',
            'engagement_rate', 'county_encoded', 'profit_encoded', 'upgrade_engagement'
        ]
        
        X = df_processed[feature_cols].values
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled
    
    def _calculate_metrics(self, df, X_processed):
        """Calculate evaluation metrics"""
        
        # Encode predicted segments
        le_segments = LabelEncoder()
        predicted_labels = le_segments.fit_transform(df['segment_target'])
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_processed, predicted_labels)
        
        metrics = {
            'silhouette_score': round(silhouette_avg, 4),
            'n_clusters': len(np.unique(predicted_labels)),
            'n_customers': len(df),
            'target_achieved': silhouette_avg > 0.6,
            'performance_tier': self._get_performance_tier(silhouette_avg)
        }
        
        # Calculate additional metrics if ground truth available
        if self.test_labels is not None:
            ground_truth = self._get_ground_truth_labels(df)
            if ground_truth is not None:
                metrics.update(self._calculate_supervised_metrics(predicted_labels, ground_truth))
        
        return metrics
    
    def _get_performance_tier(self, score):
        """Get performance tier based on silhouette score"""
        if score > 0.6:
            return "Excellent (Target Achieved)"
        elif score > 0.5:
            return "Good"
        elif score > 0.4:
            return "Fair"
        else:
            return "Needs Improvement"
    
    def _get_ground_truth_labels(self, df):
        """Get ground truth labels for supervised evaluation"""
        try:
            merged = df.merge(self.test_labels, on='customer_id', how='inner')
            le_true = LabelEncoder()
            true_labels = le_true.fit_transform(merged['segment_target_y'])
            return true_labels
        except:
            return None
    
    def _calculate_supervised_metrics(self, predicted, true_labels):
        """Calculate supervised learning metrics"""
        
        ari = adjusted_rand_score(true_labels, predicted)
        nmi = normalized_mutual_info_score(true_labels, predicted)
        
        return {
            'adjusted_rand_index': round(ari, 4),
            'normalized_mutual_info': round(nmi, 4),
            'supervised_available': True
        }

class LeaderboardSimulator:
    """Simulates a competition leaderboard"""
    
    def __init__(self, leaderboard_file="leaderboard.json"):
        self.leaderboard_file = leaderboard_file
        self.load_leaderboard()
    
    def load_leaderboard(self):
        """Load existing leaderboard or create new one"""
        try:
            with open(self.leaderboard_file, 'r') as f:
                self.leaderboard = json.load(f)
        except FileNotFoundError:
            self.leaderboard = {
                'competition': 'EABL Insights Customer Segmentation',
                'target_metric': 'Silhouette Score > 0.6',
                'submissions': []
            }
    
    def add_submission(self, team_name, metrics, submission_time=None):
        """Add a new submission to the leaderboard"""
        
        if submission_time is None:
            submission_time = datetime.now().isoformat()
        
        submission = {
            'team_name': team_name,
            'submission_time': submission_time,
            'silhouette_score': metrics['silhouette_score'],
            'n_clusters': metrics['n_clusters'],
            'n_customers': metrics['n_customers'],
            'target_achieved': metrics['target_achieved'],
            'performance_tier': metrics['performance_tier']
        }
        
        # Add supervised metrics if available
        if 'adjusted_rand_index' in metrics:
            submission.update({
                'adjusted_rand_index': metrics['adjusted_rand_index'],
                'normalized_mutual_info': metrics['normalized_mutual_info']
            })
        
        self.leaderboard['submissions'].append(submission)
        self.save_leaderboard()
        
        print(f"✅ Added submission for {team_name}")
    
    def save_leaderboard(self):
        """Save leaderboard to file"""
        with open(self.leaderboard_file, 'w') as f:
            json.dump(self.leaderboard, f, indent=2)
    
    def display_leaderboard(self, top_n=10):
        """Display the current leaderboard"""
        
        if not self.leaderboard['submissions']:
            print("📊 No submissions yet!")
            return
        
        # Sort by silhouette score
        sorted_submissions = sorted(
            self.leaderboard['submissions'],
            key=lambda x: x['silhouette_score'],
            reverse=True
        )
        
        print("\n" + "="*80)
        print("🏆 TUSKER LOYALTY CUSTOMER SEGMENTATION LEADERBOARD")
        print("="*80)
        print(f"Target: {self.leaderboard['target_metric']}")
        print("-"*80)
        
        headers = ["Rank", "Team", "Silhouette", "Clusters", "Customers", "Status"]
        print(f"{headers[0]:<6} {headers[1]:<20} {headers[2]:<12} {headers[3]:<10} {headers[4]:<12} {headers[5]}")
        print("-"*80)
        
        for i, submission in enumerate(sorted_submissions[:top_n], 1):
            status = "✅ PASS" if submission['target_achieved'] else "❌ FAIL"
            print(f"{i:<6} {submission['team_name']:<20} {submission['silhouette_score']:<12} "
                  f"{submission['n_clusters']:<10} {submission['n_customers']:<12} {status}")
        
        print("-"*80)
        
        # Show statistics
        total_subs = len(self.leaderboard['submissions'])
        passed_subs = sum(1 for s in self.leaderboard['submissions'] if s['target_achieved'])
        
        print(f"Total Submissions: {total_subs}")
        print(f"Passed Target: {passed_subs} ({passed_subs/total_subs*100:.1f}%)")
        print(f"Best Score: {sorted_submissions[0]['silhouette_score']}")
        print("="*80)

def main():
    """Main evaluation function"""
    
    print("🍺 TUSKER LOYALTY SEGMENTATION EVALUATOR")
    print("-"*50)
    
    # Initialize evaluator
    evaluator = SegmentationEvaluator()
    leaderboard = LeaderboardSimulator()
    
    # Example evaluation (you can modify this to evaluate actual submissions)
    sample_submission_path = "../data/sample_submission.csv"
    
    if os.path.exists(sample_submission_path):
        print(f"\n📝 Evaluating sample submission...")
        metrics = evaluator.evaluate_submission(sample_submission_path)
        
        if metrics:
            print(f"\n📊 EVALUATION RESULTS:")
            print(f"Silhouette Score: {metrics['silhouette_score']}")
            print(f"Performance Tier: {metrics['performance_tier']}")
            print(f"Target Achieved: {'✅ YES' if metrics['target_achieved'] else '❌ NO'}")
            print(f"Number of Clusters: {metrics['n_clusters']}")
            print(f"Customers Evaluated: {metrics['n_customers']}")
            
            # Add to leaderboard
            leaderboard.add_submission("Sample Team", metrics)
    
    # Display leaderboard
    leaderboard.display_leaderboard()
    
    print(f"\n💡 To evaluate your own submission:")
    print(f"1. Create a CSV with columns: customer_id, segment_target")
    print(f"2. Run: evaluator.evaluate_submission('your_file.csv')")
    print(f"3. Add to leaderboard: leaderboard.add_submission('Your Team', metrics)")

if __name__ == "__main__":
    main()
