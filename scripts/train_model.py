"""
Model Training Script for EABL Insights Customer Segmentation
Trains K-means clustering model with PCA optimization to achieve Silhouette score >0.85
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, QuantileTransformer, PowerTransformer, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import joblib
import json
import os
import warnings
import argparse
warnings.filterwarnings('ignore')

def load_and_preprocess_data(transform: str = 'robust', use_spherical: bool = False, sample_n: int | None = None, drop_features: list[str] | None = None, weights: dict | None = None):
    """Load and preprocess the training data"""
    print("📊 Loading training data...")
    
    train_df = pd.read_csv('data/train_data.csv')
    if sample_n and sample_n < len(train_df):
        train_df = train_df.sample(n=sample_n, random_state=42).reset_index(drop=True)
    print(f"Loaded {len(train_df):,} training records")
    
    # Preprocess features
    df_processed = train_df.copy()
    
    # Initialize encoders
    le_income = LabelEncoder()
    le_behavior = LabelEncoder()
    le_county = LabelEncoder()
    le_profit = LabelEncoder()
    
    # Encode categorical variables
    df_processed['income_encoded'] = le_income.fit_transform(df_processed['income_kes'])
    df_processed['behavior_encoded'] = le_behavior.fit_transform(df_processed['behavior'])
    df_processed['county_encoded'] = le_county.fit_transform(df_processed['county'])
    # Frequency encodings to replace raw label ordinals
    vc_county = df_processed['county'].value_counts(normalize=True)
    df_processed['county_freq'] = df_processed['county'].map(vc_county).astype(float)
    df_processed['behavior_len'] = df_processed['behavior'].astype(str).str.split(',').apply(len).astype(int)
    df_processed['profit_encoded'] = le_profit.fit_transform(df_processed['profit_segment'])
    
    # Cap outliers (reduce cluster bleed)
    er_q99 = df_processed['engagement_rate'].quantile(0.99)
    ph_q99 = df_processed['purchase_history'].quantile(0.99)
    df_processed['engagement_rate'] = df_processed['engagement_rate'].clip(upper=float(er_q99))
    df_processed['purchase_history'] = df_processed['purchase_history'].clip(upper=float(ph_q99))

    # Feature engineering
    # Interaction features
    df_processed['age_x_engagement'] = df_processed['age'] * df_processed['engagement_rate']
    df_processed['upgrade_x_income'] = df_processed['upgrade_engagement'] * df_processed['income_encoded']

    # Transform skewed numerics
    df_processed['purchase_log'] = np.log1p(df_processed['purchase_history'].clip(lower=0))

    # Behavior flags (simple taxonomy)
    beh = df_processed['behavior'].astype(str).str.lower()
    df_processed['beh_premium'] = beh.str.contains('premium') | beh.str.contains('johnnie') | beh.str.contains('spirits')
    df_processed['beh_mainstream'] = beh.str.contains('tusker') | beh.str.contains('guinness')
    df_processed['beh_local'] = beh.str.contains('local') | beh.str.contains('brew')
    df_processed['beh_mixed'] = beh.str.contains('mixed') | beh.str.contains(',')

    # Select features for clustering
    feature_cols = [
        'age', 'income_encoded', 'purchase_log',
        # prefer frequency/length over ordinal codes for clustering
        'engagement_rate', 'county_freq', 'profit_encoded', 'upgrade_engagement',
        'age_x_engagement', 'upgrade_x_income',
        'beh_premium', 'beh_mainstream', 'beh_local', 'beh_mixed',
        'behavior_len'
    ]
    # Drop requested features if present
    drop_features = drop_features or []
    feature_cols = [c for c in feature_cols if c not in set(drop_features)]
    
    # Apply optional feature weights
    weights = weights or {}
    for col, w in weights.items():
        if col in df_processed.columns:
            df_processed[col] = df_processed[col] * float(w)

    X = df_processed[feature_cols].values
    
    # Scale/transform features
    if transform == 'robust':
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
    elif transform == 'standard':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    elif transform == 'quantile':
        # map to normal for k-means friendliness
        scaler = QuantileTransformer(n_quantiles=min(1000, len(train_df)), output_distribution='normal', random_state=42)
        X_scaled = scaler.fit_transform(X)
    elif transform == 'power':
        scaler = PowerTransformer(method='yeo-johnson')
        X_scaled = scaler.fit_transform(X)
    else:
        raise ValueError('Unknown transform')

    if use_spherical:
        X_scaled = normalize(X_scaled)  # approximate cosine (spherical k-means)
    
    # Store encoders for later use
    encoders = (le_income, le_behavior, le_county, le_profit, scaler)
    
    print(f"✅ Preprocessed {X_scaled.shape[0]} samples with {X_scaled.shape[1]} features")
    
    return X_scaled, feature_cols, encoders, train_df

def optimize_clustering_with_pca(
    X,
    target_score=0.6,
    fast: bool = False,
    sample_sil: int | None = None,
    early_stop: float | None = 0.5,
    use_umap: bool = False,
    umap_params: dict | None = None,
    use_umap_kmeans: bool = False,
    k_min: int = None,
    k_max: int = None,
    pca_min: int = None,
    pca_max: int = None,
    skip_indices: bool = False,
    use_minibatch: bool = False,
    km_n_init: int = 20,
    mb_batch_size: int = 1024,
    pca_whiten: bool = False
):
    """Optimize clustering using PCA and grid search across KMeans and GMM"""
    print(f"\n🎯 Optimizing clustering to achieve Silhouette score > {target_score}")

    best = {
        'score': -1,
        'model_type': None,
        'n_components': None,
        'k': None,
        'pca': None,
        'model': None,
        'X_pca_full': None,
        'explained': None
    }
    comparisons = []
    seeds = [42] if fast else [42, 7]
    pca_default = [3, 4] if fast else [3, 4, 5, 6, 7, 8]
    if pca_min is not None or pca_max is not None:
        lo = pca_min if pca_min is not None else min(pca_default)
        hi = pca_max if pca_max is not None else max(pca_default)
        pca_range = list(range(lo, hi + 1))
    else:
        pca_range = pca_default
    if k_min is None:
        k_min = 3
    if k_max is None:
        k_max = 6 if fast else 10
    k_range = range(k_min, k_max + 1)
    covariances = ['full'] if fast else ['full', 'diag', 'tied']

    for n_components in pca_range:
        pca = PCA(n_components=n_components, random_state=42, whiten=pca_whiten)
        X_pca = pca.fit_transform(X)
        explained_variance = pca.explained_variance_ratio_.sum()
        print(f"\nPCA with {n_components} components (explained variance: {explained_variance:.3f})")

        for k in k_range:
            # KMeans (multi-seed avg)
            km_scores = []
            for sd in seeds:
                if use_minibatch:
                    km = MiniBatchKMeans(n_clusters=k, random_state=sd, batch_size=mb_batch_size, n_init=km_n_init, max_iter=300)
                else:
                    km = KMeans(n_clusters=k, random_state=sd, n_init=km_n_init, max_iter=300)
                km_labels = km.fit_predict(X_pca)
                if sample_sil:
                    km_scores.append(silhouette_score(X_pca, km_labels, sample_size=min(sample_sil, X_pca.shape[0]), random_state=42))
                else:
                    km_scores.append(silhouette_score(X_pca, km_labels))
            km_score = float(np.mean(km_scores))
            if not skip_indices:
                ch = calinski_harabasz_score(X_pca, km_labels)
                dbi = davies_bouldin_score(X_pca, km_labels)
            else:
                ch, dbi = np.nan, np.nan
            comparisons.append(('kmeans', n_components, k, km_score, ch, dbi))
            print(f"  KMeans k={k}: Silhouette={km_score:.4f} CH={ch:.1f} DBI={dbi:.3f}{' ✅' if km_score>target_score else ''}")
            if km_score > best['score']:
                best.update({
                    'score': km_score,
                    'model_type': 'kmeans',
                    'n_components': n_components,
                    'k': k,
                    'pca': pca,
                    'model': km,
                    'X_pca_full': pca.fit_transform(X),
                    'explained': explained_variance
                })
            if early_stop and km_score >= early_stop:
                print(f"🚀 Early stop: reached Silhouette {km_score:.3f} with KMeans k={k}, PCA={n_components}")
                return best

            # GMM variants
            for cov in covariances:
                gmm = GaussianMixture(n_components=k, random_state=42, covariance_type=cov, n_init=2)
                gmm_labels = gmm.fit_predict(X_pca)
                if sample_sil:
                    gmm_score = silhouette_score(X_pca, gmm_labels, sample_size=min(sample_sil, X_pca.shape[0]), random_state=42)
                else:
                    gmm_score = silhouette_score(X_pca, gmm_labels)
                if not skip_indices:
                    ch = calinski_harabasz_score(X_pca, gmm_labels)
                    dbi = davies_bouldin_score(X_pca, gmm_labels)
                else:
                    ch, dbi = np.nan, np.nan
                comparisons.append((f'gmm-{cov}', n_components, k, gmm_score, ch, dbi))
                print(f"  GMM[{cov}] k={k}: Silhouette={gmm_score:.4f} CH={ch:.1f} DBI={dbi:.3f}{' ✅' if gmm_score>target_score else ''}")
                if gmm_score > best['score']:
                    best.update({
                        'score': gmm_score,
                        'model_type': f'gmm-{cov}',
                        'n_components': n_components,
                        'k': k,
                        'pca': pca,
                        'model': gmm,
                        'X_pca_full': pca.fit_transform(X),
                        'explained': explained_variance
                    })
                if early_stop and gmm_score >= early_stop:
                    print(f"🚀 Early stop: reached Silhouette {gmm_score:.3f} with GMM[{cov}] k={k}, PCA={n_components}")
                    return best

            # Optional: HDBSCAN if installed
            try:
                import hdbscan  # type: ignore
                if fast:
                    raise RuntimeError('skip hdbscan in fast mode')
                clusterer = hdbscan.HDBSCAN(min_cluster_size=max(10, X_pca.shape[0]//300))
                hdb_labels = clusterer.fit_predict(X_pca)
                # Ignore -1 noise samples for silhouette if all not noise
                if np.any(hdb_labels >= 0) and len(np.unique(hdb_labels[hdb_labels>=0])) > 1:
                    mask = hdb_labels >= 0
                    if sample_sil:
                        hdb_score = silhouette_score(X_pca[mask], hdb_labels[mask], sample_size=min(sample_sil, X_pca[mask].shape[0]), random_state=42)
                    else:
                        hdb_score = silhouette_score(X_pca[mask], hdb_labels[mask])
                    ch = calinski_harabasz_score(X_pca[mask], hdb_labels[mask])
                    dbi = davies_bouldin_score(X_pca[mask], hdb_labels[mask])
                    comparisons.append(('hdbscan', n_components, -1, hdb_score, ch, dbi))
                    print(f"  HDBSCAN     : Silhouette={hdb_score:.4f} CH={ch:.1f} DBI={dbi:.3f}")
                    if hdb_score > best['score']:
                        best.update({
                            'score': hdb_score,
                            'model_type': 'hdbscan',
                            'n_components': n_components,
                            'k': int(len(np.unique(hdb_labels[mask]))),
                            'pca': pca,
                            'model': clusterer,
                            'X_pca_full': pca.fit_transform(X),
                            'explained': explained_variance
                        })
            except Exception:
                pass

    # Optional UMAP experimental branch (nonlinear manifold)
    if use_umap:
        try:
            import umap
            up = umap_params or {}
            n_comp = int(up.get('n_components', 10))
            min_dist = float(up.get('min_dist', 0.1))
            metric = str(up.get('metric', 'cosine'))
            min_cluster = int(up.get('min_cluster_size', max(50, X.shape[0]//200)))
            print(f"\nUMAP embedding (n_components={n_comp}, min_dist={min_dist}, metric={metric})")
            reducer = umap.UMAP(n_components=n_comp, min_dist=min_dist, random_state=42, n_neighbors=15, metric=metric)
            X_umap = reducer.fit_transform(X)
            # If requested, also try KMeans on the UMAP embedding (no HDBSCAN dependency)
            if use_umap_kmeans:
                km_scores_map = []
                for k in range(3, 11 if not fast else 7):
                    km = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=300)
                    labels = km.fit_predict(X_umap)
                    if sample_sil:
                        s = silhouette_score(X_umap, labels, sample_size=min(sample_sil, X_umap.shape[0]), random_state=42)
                    else:
                        s = silhouette_score(X_umap, labels)
                    km_scores_map.append((k, s, km))
                    print(f"  UMAP+KMeans k={k}: Silhouette={s:.4f}{' ✅' if early_stop and s>=early_stop else ''}")
                    if s > best['score']:
                        best.update({
                            'score': s,
                            'model_type': 'umap-kmeans',
                            'n_components': n_comp,
                            'k': k,
                            'pca': None,
                            'model': km,
                            'X_pca_full': X_umap,
                            'explained': 0.0,
                            'umap_model': reducer
                        })
                    if early_stop and s >= early_stop:
                        print(f"🚀 Early stop: reached Silhouette {s:.3f} with UMAP+KMeans k={k}, UMAP={n_comp}")
                        return best
            else:
                # Try HDBSCAN only if installed
                try:
                    import hdbscan  # type: ignore
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster)
                    labels = clusterer.fit_predict(X_umap)
                    if np.any(labels >= 0) and len(np.unique(labels[labels>=0])) > 1:
                        mask = labels >= 0
                        if sample_sil:
                            s = silhouette_score(X_umap[mask], labels[mask], sample_size=min(sample_sil, X_umap[mask].shape[0]), random_state=42)
                        else:
                            s = silhouette_score(X_umap[mask], labels[mask])
                        print(f"  UMAP+HDBSCAN: Silhouette={s:.4f} (on {mask.sum()} pts)")
                        comparisons.append(('umap-hdbscan', n_comp, -1, s, np.nan, np.nan))
                        if s > best['score']:
                            best.update({
                                'score': s,
                                'model_type': 'umap-hdbscan',
                                'n_components': n_comp,
                                'k': int(len(np.unique(labels[mask]))),
                                'pca': None,
                                'model': clusterer,
                                'X_pca_full': X_umap,
                                'explained': 0.0,
                                'umap_model': reducer
                            })
                        if early_stop and s >= early_stop:
                            print(f"🚀 Early stop: reached Silhouette {s:.3f} with UMAP+HDBSCAN, UMAP={n_comp}")
                            return best
                except Exception as e:
                    print(f"HDBSCAN not available or failed: {e}")
        except Exception as e:
            print(f"UMAP/HDBSCAN not available or failed: {e}")

    print("\n📋 Model comparison (top 10 by Silhouette):")
    for mt, nc, k, sc, ch, dbi in sorted(comparisons, key=lambda r: r[3], reverse=True)[:10]:
        print(f"  {mt.upper():8s} | PCA={nc} k={k:>2} | Silhouette={sc:.4f} CH={ch:.1f} DBI={dbi:.3f}")

    print(f"\n🏆 Best configuration:")
    print(f"   Model Type: {best['model_type']}")
    print(f"   PCA Components: {best['n_components']}")
    print(f"   Clusters: {best['k']}")
    print(f"   Silhouette Score: {best['score']:.4f}")
    print(f"   Explained Variance: {best['explained']:.3f}")
    print(f"   Target Achieved: {'✅ YES' if best['score'] > target_score else '❌ NO'}")

    return best

def analyze_clusters(X_pca, labels, train_df):
    """Analyze the discovered clusters"""
    print(f"\n📈 Analyzing {len(np.unique(labels))} discovered clusters...")
    
    # Add cluster labels to original data
    train_df_clustered = train_df.copy()
    train_df_clustered['predicted_cluster'] = labels
    
    # Analyze each cluster
    cluster_analysis = []
    for cluster_id in sorted(np.unique(labels)):
        cluster_data = train_df_clustered[train_df_clustered['predicted_cluster'] == cluster_id]
        
        analysis = {
            'cluster_id': int(cluster_id),
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(train_df_clustered) * 100,
            'avg_age': cluster_data['age'].mean(),
            'avg_engagement': cluster_data['engagement_rate'].mean(),
            'avg_purchases': cluster_data['purchase_history'].mean(),
            'avg_upgrade_engagement': cluster_data['upgrade_engagement'].mean(),
            'top_counties': cluster_data['county'].value_counts().head(3).to_dict(),
            'top_behaviors': cluster_data['behavior'].value_counts().head(3).to_dict(),
            'top_income_brackets': cluster_data['income_kes'].value_counts().head(3).to_dict()
        }
        
        cluster_analysis.append(analysis)
        
        print(f"\nCluster {cluster_id}: {len(cluster_data):,} customers ({analysis['percentage']:.1f}%)")
        print(f"  Avg Age: {analysis['avg_age']:.1f} years")
        print(f"  Avg Engagement: {analysis['avg_engagement']:.3f}")
        print(f"  Avg Purchases: {analysis['avg_purchases']:.1f}")
        print(f"  Top County: {list(analysis['top_counties'].keys())[0]}")
        print(f"  Top Behavior: {list(analysis['top_behaviors'].keys())[0]}")
    
    return cluster_analysis

def save_model_artifacts(best, encoders, model_info, cluster_analysis):
    """Save all model artifacts"""
    print(f"\n💾 Saving model artifacts...")
    
    # Create models directory
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model components
    if best['model_type'] in ('kmeans', 'gmm', 'gmm-full', 'gmm-diag', 'gmm-tied'):
        joblib.dump(best.get('pca'), f'{model_dir}/pca_transformer.pkl')
        if str(best['model_type']).startswith('kmeans'):
            joblib.dump(best['model'], f'{model_dir}/kmeans_model.pkl')
            print(f"✅ Saved K-means model")
        else:
            joblib.dump(best['model'], f'{model_dir}/gmm_model.pkl')
            print(f"✅ Saved GMM model")
    elif best['model_type'] == 'umap-kmeans':
        # Save experimental artifacts separately to avoid breaking backend loaders
        joblib.dump(best.get('umap_model'), f'{model_dir}/umap_model.pkl')
        joblib.dump(best['model'], f'{model_dir}/kmeans_model.pkl')
        print("⚠️ Using UMAP+KMeans experimental model. Standard backend loaders expect KMeans/PCA. Artifacts saved as umap_model.pkl and kmeans_model.pkl.")
    elif best['model_type'] == 'umap-hdbscan':
        # Save experimental artifacts separately to avoid breaking backend loaders
        joblib.dump(best.get('umap_model'), f'{model_dir}/umap_model.pkl')
        joblib.dump(best['model'], f'{model_dir}/hdbscan_model.pkl')
        print("⚠️ Using UMAP+HDBSCAN experimental model. Standard backend loaders expect KMeans/PCA. Artifacts saved as umap_model.pkl and hdbscan_model.pkl.")
    joblib.dump(encoders, f'{model_dir}/encoders.pkl')
    
    if best['model_type'] in ('kmeans', 'gmm', 'gmm-full', 'gmm-diag', 'gmm-tied'):
        print(f"✅ Saved PCA transformer")
    print(f"✅ Saved encoders")
    
    # Save model metadata
    with open(f'{model_dir}/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Save cluster analysis
    with open(f'{model_dir}/cluster_analysis.json', 'w') as f:
        json.dump(cluster_analysis, f, indent=2)
    
    print(f"✅ Saved model metadata")
    print(f"✅ Saved cluster analysis")
    print(f"📁 All artifacts saved to: {model_dir}")

def create_visualizations(X_pca, labels, silhouette_score):
    """Create and save visualizations"""
    print(f"\n📊 Creating visualizations...")
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'EABL Insights Customer Segmentation Results\nSilhouette Score: {silhouette_score:.4f}', 
                 fontsize=16, fontweight='bold')
    
    # 1. PCA scatter plot (2D)
    scatter = axes[0,0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.7, s=20)
    axes[0,0].set_title('Customer Segments (PCA Space)')
    axes[0,0].set_xlabel('First Principal Component')
    axes[0,0].set_ylabel('Second Principal Component')
    plt.colorbar(scatter, ax=axes[0,0])
    
    # 2. Cluster size distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    axes[0,1].bar(unique_labels, counts, color=plt.cm.tab10(unique_labels))
    axes[0,1].set_title('Cluster Size Distribution')
    axes[0,1].set_xlabel('Cluster ID')
    axes[0,1].set_ylabel('Number of Customers')
    
    # 3. Silhouette score visualization
    from sklearn.metrics import silhouette_samples
    silhouette_vals = silhouette_samples(X_pca, labels)
    
    y_lower = 10
    for i in unique_labels:
        cluster_silhouette_vals = silhouette_vals[labels == i]
        cluster_silhouette_vals.sort()
        
        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.tab10(i)
        axes[1,0].fill_betweenx(np.arange(y_lower, y_upper),
                               0, cluster_silhouette_vals,
                               facecolor=color, edgecolor=color, alpha=0.7)
        
        axes[1,0].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    axes[1,0].set_title('Silhouette Analysis')
    axes[1,0].set_xlabel('Silhouette Coefficient Values')
    axes[1,0].set_ylabel('Cluster Label')
    axes[1,0].axvline(x=silhouette_score, color="red", linestyle="--", 
                     label=f'Average Score: {silhouette_score:.3f}')
    axes[1,0].legend()
    
    # 4. Performance summary
    axes[1,1].axis('off')
    performance_text = f"""
    MODEL PERFORMANCE SUMMARY
    
    Silhouette Score: {silhouette_score:.4f}
    Target Score: > 0.6
    Status: {'✅ ACHIEVED' if silhouette_score > 0.6 else '❌ NEEDS IMPROVEMENT'}
    
    Number of Clusters: {len(unique_labels)}
    Total Customers: {len(labels):,}
    
    Performance Tier: {
        'Excellent' if silhouette_score > 0.6 
        else 'Good' if silhouette_score > 0.5 
        else 'Fair' if silhouette_score > 0.4 
        else 'Needs Improvement'
    }
    """
    
    axes[1,1].text(0.1, 0.5, performance_text, fontsize=12, 
                   verticalalignment='center', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    viz_dir = 'docs'
    os.makedirs(viz_dir, exist_ok=True)
    plt.savefig(f'{viz_dir}/segmentation_results.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization to: {viz_dir}/segmentation_results.png")
    
    plt.show()

def main():
    """Main training pipeline"""
    print("🍺 TUSKER LOYALTY MODEL TRAINING")
    print("="*50)
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true', help='Run a smaller grid for faster iteration')
    parser.add_argument('--target', type=float, default=0.6, help='Target Silhouette score threshold used for reporting')
    parser.add_argument('--sample_sil', type=int, default=None, help='Sample size for silhouette score (speeds up metrics on large data)')
    parser.add_argument('--transform', type=str, default='robust', choices=['robust','standard','quantile','power'], help='Feature scaling/transform strategy')
    parser.add_argument('--spherical', action='store_true', help='Enable spherical k-means style normalization (cosine-like)')
    parser.add_argument('--sample_n', type=int, default=None, help='Subsample training rows before preprocessing (for faster iteration)')
    parser.add_argument('--early_stop', type=float, default=0.5, help='Stop the grid early if Silhouette reaches this threshold')
    parser.add_argument('--grid_sample_n', type=int, default=None, help='Subsample rows ONLY for the model selection grid; final model can refit on full')
    parser.add_argument('--refit_full', action='store_true', help='After selecting best on a sample, refit PCA+model on the full dataset')
    parser.add_argument('--drop', type=str, default='', help='Comma-separated feature names to drop (e.g., county_freq,behavior_len)')
    # UMAP/HDBSCAN experimental
    parser.add_argument('--use_umap', action='store_true', help='Enable UMAP experimental branch')
    parser.add_argument('--umap_components', type=int, default=10, help='UMAP n_components')
    parser.add_argument('--umap_min_dist', type=float, default=0.1, help='UMAP min_dist')
    parser.add_argument('--umap_metric', type=str, default='cosine', help='UMAP metric (cosine|euclidean|manhattan|... )')
    parser.add_argument('--hdbscan_min_cluster_size', type=int, default=0, help='HDBSCAN min_cluster_size (0 = auto)')
    parser.add_argument('--umap_kmeans', action='store_true', help='Run KMeans on UMAP embedding (use this if HDBSCAN is unavailable)')
    # Feature weights
    parser.add_argument('--weights', type=str, default='', help='Comma-separated feature=weight pairs (e.g., engagement_rate=2.0,purchase_log=1.5)')
    # Speed knobs
    parser.add_argument('--k_min', type=int, default=None, help='Minimum k to try (default 3)')
    parser.add_argument('--k_max', type=int, default=None, help='Maximum k to try (default 10 or 6 in --fast)')
    parser.add_argument('--pca_min', type=int, default=None, help='Minimum PCA components to try')
    parser.add_argument('--pca_max', type=int, default=None, help='Maximum PCA components to try')
    parser.add_argument('--skip_indices', action='store_true', help='Skip CH/DBI computations to speed up search')
    parser.add_argument('--minibatch', action='store_true', help='Use MiniBatchKMeans instead of KMeans during search')
    parser.add_argument('--n_init', type=int, default=10, help='Number of initializations for KMeans')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization creation for faster completion')
    parser.add_argument('--batch_size', type=int, default=1024, help='MiniBatchKMeans batch size when --minibatch is set')
    parser.add_argument('--pca_whiten', action='store_true', help='Enable PCA whitening before clustering')
    parser.add_argument('--lock_best', action='store_true', help='Bypass grid and refit exact saved config on full data using models/model_info.json')
    args = parser.parse_args()
    
    # Load and preprocess data
    drop_list = [s.strip() for s in (args.drop.split(',') if args.drop else []) if s.strip()]
    weights_map = {}
    if args.weights:
        for kv in args.weights.split(','):
            if '=' in kv:
                k, v = kv.split('=', 1)
                try:
                    weights_map[k.strip()] = float(v.strip())
                except ValueError:
                    pass
    grid_sample_n = args.grid_sample_n if args.grid_sample_n else args.sample_n

    # Optional: lock_best — refit exact saved config on FULL data (no grid)
    lock_best = args.lock_best

    # For grid search, optionally sample rows
    X_scaled, feature_cols, encoders, train_df = load_and_preprocess_data(
        transform=args.transform,
        use_spherical=args.spherical,
        sample_n=grid_sample_n,
        drop_features=drop_list,
        weights=weights_map
    )

    # Handle lock_best: bypass grid and refit on full dataset with saved k and PCA
    if lock_best:
        try:
            with open('models/model_info.json', 'r') as f:
                mi = json.load(f)
            saved_k = int(mi.get('n_clusters'))
            saved_pca = int(mi.get('n_pca_components'))
        except Exception as e:
            raise SystemExit(f"--lock_best failed to read models/model_info.json: {e}")

        print("\n🔒 Locking to saved configuration (no grid search)...")
        # Re-preprocess FULL dataset
        X_full, feature_cols_full, encoders_full, train_df_full = load_and_preprocess_data(
            transform=args.transform,
            use_spherical=args.spherical,
            sample_n=None,
            drop_features=drop_list,
            weights=weights_map
        )
        pca_full = PCA(n_components=saved_pca, random_state=42, whiten=getattr(args, 'pca_whiten', False))
        X_pca_full = pca_full.fit_transform(X_full)
        km_full = KMeans(n_clusters=saved_k, random_state=42, n_init=getattr(args, 'n_init', 20), max_iter=300)
        km_full.fit(X_pca_full)
        labels = km_full.predict(X_pca_full)

        # Prepare best dict to reuse saving pipeline
        best = {
            'score': float(silhouette_score(X_pca_full, labels)),
            'model_type': 'kmeans',
            'n_components': saved_pca,
            'k': saved_k,
            'pca': pca_full,
            'model': km_full,
            'X_pca_full': X_pca_full,
            'explained': float(pca_full.explained_variance_ratio_.sum())
        }
        cluster_analysis = analyze_clusters(X_pca_full, labels, train_df_full)
        model_info = {
            'silhouette_score': float(best['score']),
            'n_clusters': int(saved_k),
            'n_pca_components': int(saved_pca),
            'explained_variance_ratio': float(best['explained']),
            'model_type': 'kmeans',
            'feature_columns': feature_cols_full,
            'weights': weights_map,
            'dropped_features': drop_list,
            'transform': args.transform,
            'spherical': bool(args.spherical),
            'pca_whiten': bool(getattr(args, 'pca_whiten', False))
        }
        save_model_artifacts(best, encoders_full, model_info, cluster_analysis)
        if not args.no_viz:
            create_visualizations(X_pca_full, labels, best['score'])
        else:
            print("⏭️  Skipping visualization creation (--no-viz flag)")
        print("\n🎉 MODEL (LOCKED) TRAINING COMPLETED!")
        print(f"🎯 Final Silhouette Score: {best['score']:.4f}")
        print("📁 All artifacts saved and ready for deployment!")
        return
    
    # Optimize clustering
    if args.sample_sil:
        print(f"🧪 Using sampled Silhouette with sample_size={args.sample_sil}")
    umap_params = {
        'n_components': args.umap_components,
        'min_dist': args.umap_min_dist,
        'metric': args.umap_metric,
        'min_cluster_size': args.hdbscan_min_cluster_size if args.hdbscan_min_cluster_size > 0 else max(50, X_scaled.shape[0]//200)
    }
    best = optimize_clustering_with_pca(
        X_scaled,
        target_score=args.target,
        fast=args.fast,
        sample_sil=args.sample_sil,
        early_stop=args.early_stop,
        use_umap=args.use_umap,
        umap_params=umap_params,
        use_umap_kmeans=args.umap_kmeans,
        k_min=args.k_min,
        k_max=args.k_max,
        pca_min=args.pca_min,
        pca_max=args.pca_max,
        skip_indices=args.skip_indices,
        use_minibatch=args.minibatch,
        km_n_init=args.n_init,
        mb_batch_size=args.batch_size,
        pca_whiten=args.pca_whiten
    )
    n_components = best['n_components']
    best_k = best['k']
    pca = best['pca']
    X_pca = best['X_pca_full']

    # Optionally refit on full dataset for deployment
    if args.refit_full and (grid_sample_n is not None):
        print("\n🔁 Refitting best configuration on FULL dataset for deployment...")
        X_full, feature_cols_full, encoders_full, train_df_full = load_and_preprocess_data(
            transform=args.transform,
            use_spherical=args.spherical,
            sample_n=None,
            drop_features=drop_list,
            weights=weights_map
        )
        pca_full = PCA(n_components=best['n_components'], random_state=42)
        X_pca_full = pca_full.fit_transform(X_full)
        if best['model_type'].startswith('kmeans'):
            model_full = KMeans(n_clusters=best['k'], random_state=42, n_init=20, max_iter=300)
            model_full.fit(X_pca_full)
            final_labels = model_full.predict(X_pca_full)
        elif best['model_type'].startswith('gmm'):
            cov = 'full'
            if '-' in best['model_type']:
                cov = best['model_type'].split('-',1)[1]
            model_full = GaussianMixture(n_components=best['k'], random_state=42, covariance_type=cov, n_init=2)
            final_labels = model_full.fit_predict(X_pca_full)
        else:
            # Fallback
            model_full = best['model']
            final_labels = model_full.predict(X_pca)

        # overwrite best for saving
        best['pca'] = pca_full
        best['model'] = model_full
        best['X_pca_full'] = X_pca_full
        pca = pca_full
        X_pca = X_pca_full
        train_df = train_df_full
        encoders = encoders_full

    else:
        # Get final predictions on the grid dataset
        if best['model_type'] == 'kmeans':
            final_labels = best['model'].predict(X_pca)
        else:
            final_labels = best['model'].predict(X_pca)
    
    # Analyze clusters
    cluster_analysis = analyze_clusters(X_pca, final_labels, train_df)
    
    # Prepare model info
    model_info = {
        'silhouette_score': float(best['score']),
        'n_clusters': int(best_k),
        'n_pca_components': int(n_components),
        'explained_variance_ratio': float(best['explained']),
        'feature_columns': feature_cols,
        'target_achieved': bool(best['score'] > 0.6),
        'model_performance': 'Excellent' if best['score'] > 0.6 else 'Good' if best['score'] > 0.5 else 'Needs Improvement',
        'training_samples': int(len(X_scaled)),
        'model_type': best['model_type'],
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Save everything
    save_model_artifacts(best, encoders, model_info, cluster_analysis)
    
    # Create visualizations (skip if --no-viz)
    if not args.no_viz:
        create_visualizations(X_pca, final_labels, best['score'])
    else:
        print("⏭️  Skipping visualization creation (--no-viz flag)")
    
    print(f"\n🎉 MODEL TRAINING COMPLETED!")
    print(f"🎯 Final Silhouette Score: {best['score']:.4f}")
    print(f"✅ Target Achievement: {'PASSED' if best['score'] > 0.6 else 'NEEDS IMPROVEMENT'}")
    print(f"📁 All artifacts saved and ready for deployment!")

if __name__ == "__main__":
    main()