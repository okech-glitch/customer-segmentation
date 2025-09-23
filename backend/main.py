"""
FastAPI Backend for EABL Insights Customer Segmentation Challenge
Provides endpoints for data upload, model inference, and visualizations
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import numpy as np
import json
import asyncio
import time
import io
import os
from typing import List, Dict, Any
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import json
import requests

app = FastAPI(
    title="EABL Insights API",
    description="Customer Segmentation API for Kenyan Market Analysis (Target: Silhouette >0.6)",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model components
model_components = {
    'pca': None,
    'umap': None,  # optional
    'kmeans': None,
    'encoders': None,
    'info': {}
}
model_loaded = False
_model_info_mtime = 0.0
last_results_df = None  # enriched DataFrame from last upload (with predictions)

def load_model_components():
    """Load trained model components"""
    global model_components, model_loaded
    
    try:
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

        # Load model info first (to decide which artifacts to expect)
        with open(f'{model_dir}/model_info.json', 'r') as f:
            info = json.load(f)
        model_components['info'] = info

        # Try to load UMAP+KMeans artifacts if model_type indicates so
        umap_loaded = False
        if str(info.get('model_type', '')).startswith('umap-kmeans'):
            try:
                model_components['umap'] = joblib.load(f'{model_dir}/umap_model.pkl')
                model_components['kmeans'] = joblib.load(f'{model_dir}/kmeans_model.pkl')
                umap_loaded = True
                print("✅ UMAP+KMeans artifacts loaded")
            except Exception as ue:
                print(f"⚠️ Could not load UMAP+KMeans artifacts, will fallback to PCA+KMeans: {ue}")

        if not umap_loaded:
            # Fallback to PCA+KMeans
            model_components['pca'] = joblib.load(f'{model_dir}/pca_transformer.pkl')
            model_components['kmeans'] = joblib.load(f'{model_dir}/kmeans_model.pkl')
            print("✅ PCA+KMeans artifacts loaded")

        # Load encoders/scaler
        model_components['encoders'] = joblib.load(f'{model_dir}/encoders.pkl')
        
        model_loaded = True
        print("✅ Model components loaded successfully")
        
    except Exception as e:
        print(f"⚠️ Could not load model components: {e}")
        model_loaded = False

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    load_model_components()
    # Start background watcher to auto-reload model artifacts when they change
    async def _watch_models_dir():
        global _model_info_mtime
        model_info_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model_info.json')
        # Normalize path
        model_info_path = os.path.abspath(model_info_path)
        while True:
            try:
                if os.path.exists(model_info_path):
                    mtime = os.path.getmtime(model_info_path)
                    if mtime != _model_info_mtime:
                        _model_info_mtime = mtime
                        print("🔄 Detected model artifacts change. Reloading...")
                        load_model_components()
                await asyncio.sleep(3)
            except Exception as e:
                # Do not crash the server due to watcher errors
                print(f"Watcher error: {e}")
                await asyncio.sleep(5)
    asyncio.create_task(_watch_models_dir())

@app.get("/")
async def root():
    """API health check and information"""
    return {
        "message": "🍺 EABL Insights Customer Segmentation API",
        "version": "1.0.0",
        "status": "active",
        "model_loaded": model_loaded,
        "endpoints": [
            "/upload - Upload CSV for segmentation",
            "/generate-data - Generate sample data",
            "/visualize - Create cluster visualizations",
            "/model-info - Get model performance metrics"
        ]
    }

@app.get("/model-info")
async def get_model_info():
    """Get information about the trained model"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return model_components['info']

@app.post("/reload-model")
async def reload_model():
    """Reload trained model artifacts from disk. Use after training completes."""
    try:
        load_model_components()
        if not model_loaded:
            raise HTTPException(status_code=500, detail="Model reload failed")
        return {"status": "success", "model_loaded": True, "info": model_components['info']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")

@app.post("/generate-data")
async def generate_sample_data(n_samples: int = 1000):
    """Generate sample data for testing"""
    try:
        import random
        import numpy as np
        
        # Set seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # Kenyan counties
        counties = ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret', 'Thika', 
                   'Malindi', 'Kitale', 'Garissa', 'Kakamega']
        
        # Income brackets
        income_brackets = ["Below 20k", "20k-50k", "50k-100k", "100k-200k", "200k-500k", "Above 500k"]
        
        # Behaviors
        behaviors = ["Tusker,Vooma", "Premium Spirits", "Budget Conscious", "Social Drinker", 
                    "Occasional Drinker", "Premium Consumer", "Mixed Portfolio"]
        
        # Profit segments
        profit_segments = ["High-Spirits", "Premium-Beer", "Volume-Driver", "Emerging-Consumer", 
                          "Price-Sensitive"]
        
        # Generate data
        data = []
        for i in range(n_samples):
            record = {
                'customer_id': 1000 + i,
                'age': random.randint(18, 70),
                'income_kes': random.choice(income_brackets),
                'purchase_history': random.randint(1, 50),
                'behavior': random.choice(behaviors),
                'engagement_rate': round(random.uniform(0.1, 0.95), 3),
                'county': random.choice(counties),
                'profit_segment': random.choice(profit_segments),
                'upgrade_engagement': round(random.uniform(0.8, 2.0), 3),
                'segment_target': 'Urban Youth' if i % 4 == 0 else 'Budget Conscious'
            }
            data.append(record)
        
        df = pd.DataFrame(data)
        
        # Convert to JSON
        data_json = df.to_dict('records')
        
        # persist for map endpoint
        global last_results_df
        last_results_df = df.copy()

        return {
            "status": "success",
            "message": f"Generated {n_samples} sample records",
            "data": data_json[:100],  # Return first 100 for preview
            "total_records": len(data_json),
            "columns": list(df.columns)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data generation failed: {str(e)}")

def preprocess_data(df: pd.DataFrame):
    """Preprocess data for clustering using saved encoders and training feature_columns.
    Reconstructs engineered features to match training pipeline.
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df_processed = df.copy()
        le_income, le_behavior, le_county, le_profit, scaler = model_components['encoders']
        info = model_components.get('info', {})
        feature_cols = info.get('feature_columns') or []

        # Basic encodings
        df_processed['income_encoded'] = le_income.transform(df_processed['income_kes'])
        df_processed['behavior_encoded'] = le_behavior.transform(df_processed['behavior'])
        df_processed['county_encoded'] = le_county.transform(df_processed['county'])
        df_processed['profit_encoded'] = le_profit.transform(df_processed['profit_segment'])

        # Outlier caps similar to training (compute from incoming batch)
        er_q99 = df_processed['engagement_rate'].quantile(0.99)
        ph_q99 = df_processed['purchase_history'].quantile(0.99)
        df_processed['engagement_rate'] = df_processed['engagement_rate'].clip(upper=float(er_q99))
        df_processed['purchase_history'] = df_processed['purchase_history'].clip(upper=float(ph_q99))

        # Engineered features
        df_processed['age_x_engagement'] = df_processed['age'] * df_processed['engagement_rate']
        df_processed['upgrade_x_income'] = df_processed['upgrade_engagement'] * df_processed['income_encoded']
        df_processed['purchase_log'] = np.log1p(df_processed['purchase_history'].clip(lower=0))

        beh = df_processed['behavior'].astype(str).str.lower()
        df_processed['beh_premium'] = beh.str.contains('premium') | beh.str.contains('johnnie') | beh.str.contains('spirits')
        df_processed['beh_mainstream'] = beh.str.contains('tusker') | beh.str.contains('guinness')
        df_processed['beh_local'] = beh.str.contains('local') | beh.str.contains('brew')
        df_processed['beh_mixed'] = beh.str.contains('mixed') | beh.str.contains(',')
        df_processed['behavior_len'] = df_processed['behavior'].astype(str).str.split(',').apply(len).astype(int)

        # County frequency (approximate with current batch)
        vc_county = df_processed['county'].value_counts(normalize=True)
        df_processed['county_freq'] = df_processed['county'].map(vc_county).astype(float)

        # Apply saved feature weights (if provided in model_info) BEFORE scaling
        weights = info.get('weights') or {}
        if isinstance(weights, dict):
            for col, w in weights.items():
                if col in df_processed.columns:
                    try:
                        df_processed[col] = df_processed[col].astype(float) * float(w)
                    except Exception:
                        pass

        # Assemble features in the same order as training
        if feature_cols:
            missing = [c for c in feature_cols if c not in df_processed.columns]
            if missing:
                raise HTTPException(status_code=400, detail=f"Missing engineered features at inference: {missing}")
            X = df_processed[feature_cols].astype(float)
        else:
            # Fallback to legacy columns
            legacy_cols = [
                'age', 'income_encoded', 'purchase_history', 'behavior_encoded',
                'engagement_rate', 'county_encoded', 'profit_encoded', 'upgrade_engagement'
            ]
            X = df_processed[legacy_cols].astype(float)

        # Scale
        X_scaled = scaler.transform(X)
        return X_scaled

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(e)}")

@app.post("/upload")
async def upload_and_segment(file: UploadFile = File(...)):
    """Upload CSV file and perform customer segmentation"""
    
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validate required columns
        required_cols = ['customer_id', 'age', 'income_kes', 'purchase_history', 
                        'behavior', 'engagement_rate', 'county', 'profit_segment', 
                        'upgrade_engagement']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Preprocess data
        X_scaled = preprocess_data(df)
        
        # Choose representation: UMAP if available, else PCA
        rep = None
        if model_components.get('umap') is not None and str(model_components['info'].get('model_type','')).startswith('umap-kmeans'):
            try:
                umap_model = model_components['umap']
                X_rep = umap_model.transform(X_scaled)
                rep = 'umap'
            except Exception as e:
                # Fallback to PCA if UMAP transform not available
                pca = model_components.get('pca')
                if pca is None:
                    raise HTTPException(status_code=500, detail=f"UMAP transform failed and PCA not available: {str(e)}")
                X_rep = pca.transform(X_scaled)
                rep = 'pca'
        else:
            pca = model_components['pca']
            X_rep = pca.transform(X_scaled)
            rep = 'pca'

        # Predict clusters
        kmeans = model_components['kmeans']
        cluster_labels = kmeans.predict(X_rep)

        # Calculate silhouette score on the used space
        silhouette_avg = silhouette_score(X_rep, cluster_labels)
        
        # Add cluster names for readability (and keep columns for later map aggregation)
        df_results = df.copy()
        df_results['predicted_cluster'] = cluster_labels
        df_results['cluster_name'] = [f"Cluster {c}" for c in cluster_labels]
        
        # persist for map endpoint
        global last_results_df
        last_results_df = df_results.copy()

        # Generate cluster summary
        cluster_summary = []
        for cluster_id in sorted(df_results['predicted_cluster'].unique()):
            cluster_data = df_results[df_results['predicted_cluster'] == cluster_id]
            
            summary = {
                "cluster_id": int(cluster_id),
                "cluster_name": cluster_data['cluster_name'].iloc[0],
                "size": len(cluster_data),
                "percentage": round(len(cluster_data) / len(df_results) * 100, 1),
                "avg_age": round(cluster_data['age'].mean(), 1),
                "avg_engagement": round(cluster_data['engagement_rate'].mean(), 3),
                "avg_purchases": round(cluster_data['purchase_history'].mean(), 1),
                "top_counties": cluster_data['county'].value_counts().head(3).to_dict(),
                "top_behaviors": cluster_data['behavior'].value_counts().head(3).to_dict()
            }
            cluster_summary.append(summary)
        
        return {
            "status": "success",
            "message": f"Successfully segmented {len(df)} customers",
            "silhouette_score": round(silhouette_avg, 4),
            "model_performance": "Excellent" if silhouette_avg > 0.6 else "Good" if silhouette_avg > 0.5 else "Needs Improvement",
            "total_customers": len(df),
            "n_clusters": len(cluster_summary),
            "cluster_summary": cluster_summary,
            # keep minimal predictions for UI, backend keeps enriched DataFrame in memory
            "predictions": df_results[['customer_id', 'predicted_cluster', 'cluster_name']].to_dict('records')
        }
    except HTTPException as he:
        # Preserve original client error (e.g., 400 for missing columns)
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")

@app.post("/visualize")
async def create_visualizations(data: Dict[str, Any]):
    """Create cluster visualizations"""
    
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Extract data from request
        df = pd.DataFrame(data['predictions'])
        
        # Create visualizations
        visualizations = {}
        
        # 1. Cluster distribution pie chart
        cluster_counts = df['cluster_name'].value_counts()
        fig_pie = px.pie(
            values=cluster_counts.values,
            names=cluster_counts.index,
            title="Customer Segment Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        visualizations['cluster_distribution'] = json.loads(fig_pie.to_json())
        
        # 2. Age vs Engagement scatter plot
        fig_scatter = px.scatter(
            df, x='age', y='engagement_rate', 
            color='cluster_name',
            title="Age vs Engagement Rate by Segment",
            labels={'age': 'Age', 'engagement_rate': 'Engagement Rate'}
        )
        visualizations['age_engagement'] = json.loads(fig_scatter.to_json())
        
        # 3. Cluster size bar chart
        fig_bar = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            title="Segment Sizes",
            labels={'x': 'Customer Segment', 'y': 'Number of Customers'},
            color=cluster_counts.values,
            color_continuous_scale='viridis'
        )
        visualizations['cluster_sizes'] = json.loads(fig_bar.to_json())
        
        return {
            "status": "success",
            "visualizations": visualizations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")


# -----------------------------
# Map Figure Endpoint (Option B)
# -----------------------------

def _load_kenya_geojson_backend() -> Dict[str, Any]:
    """Load Kenya counties GeoJSON from backend data folder or from mirrors."""
    local_paths = [
        os.path.join(os.path.dirname(__file__), 'data', 'kenya-counties.geojson'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend', 'public', 'data', 'kenya-counties.geojson'),
    ]
    for p in local_paths:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
    mirrors = [
        'https://raw.githubusercontent.com/wmgeolab/geoBoundaries/main/releaseData/gbOpen/KEN/ADM1/geoBoundaries-KEN-ADM1.geojson',
        'https://cdn.jsdelivr.net/gh/wmgeolab/geoBoundaries@main/releaseData/gbOpen/KEN/ADM1/geoBoundaries-KEN-ADM1.geojson',
        'https://raw.githubusercontent.com/statoood/kenya-geojson/master/kenya-counties.geojson'
    ]
    for url in mirrors:
        try:
            r = requests.get(url, timeout=10)
            if r.ok:
                return r.json()
        except Exception:
            continue
    raise HTTPException(status_code=503, detail='Kenya GeoJSON not available locally or from mirrors')


def _detect_county_name_prop(geojson: Dict[str, Any]) -> str:
    props = (geojson.get('features') or [{}])[0].get('properties', {})
    for k in ['COUNTY_NAM', 'COUNTY', 'ADM1_EN', 'NAME_1', 'name', 'Name']:
        if k in props:
            return k
    return next(iter(props.keys()), 'name')


@app.get("/map/figure")
async def map_figure(metric: str = 'customers', segment: str = 'All'):
    """Return a Plotly Figure JSON for Kenya county choropleth.

    metric: customers | share | engagement
    segment: 'All' or a specific cluster_name
    """
    global last_results_df
    if last_results_df is None:
        raise HTTPException(status_code=400, detail='No results available. Upload data first.')

    try:
        geojson = _load_kenya_geojson_backend()
        name_prop = _detect_county_name_prop(geojson)

        df = last_results_df.copy()
        if segment != 'All':
            df = df[df['cluster_name'] == segment]

        if 'county' not in df.columns:
            raise HTTPException(status_code=400, detail='Uploaded data lacks county column for map aggregation.')

        # Aggregate by county
        if metric == 'customers':
            agg = df.groupby('county').size().rename('value').reset_index()
        elif metric == 'engagement':
            if 'engagement_rate' not in df.columns:
                raise HTTPException(status_code=400, detail='Uploaded data lacks engagement_rate for map aggregation.')
            agg = df.groupby('county')['engagement_rate'].mean().rename('value').reset_index()
        elif metric == 'share':
            total = len(last_results_df) if segment == 'All' else len(df)
            if total == 0:
                total = 1
            counts = df.groupby('county').size()
            agg = (counts / total * 100).rename('value').reset_index()
        else:
            raise HTTPException(status_code=400, detail='Unknown metric')

        # Map to geojson feature ids
        def sanitize(s: str) -> str:
            return ''.join(ch for ch in s.lower().strip() if ch.isalnum())

        county_to_value = {sanitize(r['county']): float(r['value']) for _, r in agg.iterrows()}
        feature_ids = []
        z_values = []
        for f in geojson.get('features', []):
            nm = f.get('properties', {}).get(name_prop, '')
            fid = sanitize(str(nm))
            feature_ids.append(nm)
            z_values.append(county_to_value.get(fid, 0.0))

        # Build figure
        colorscale = [
            [0, '#EEF2FF'], [0.2, '#C7D2FE'], [0.4, '#A5B4FC'], [0.6, '#818CF8'], [0.8, '#6366F1'], [1, '#4F46E5']
        ]
        title = (
            'Customers per County' if metric == 'customers'
            else 'Segment Share (%) per County' if metric == 'share'
            else 'Average Engagement per County'
        )

        fig = go.Figure(go.Choropleth(
            geojson=geojson,
            locations=feature_ids,
            z=z_values,
            featureidkey=f'properties.{name_prop}',
            colorscale=colorscale,
            marker_line_width=0,
            colorbar_title=('Customers' if metric == 'customers' else 'Share %' if metric == 'share' else 'Engagement')
        ))
        # Use geo layout focused on Kenya extents
        fig.update_layout(
            title=dict(text=title, x=0),
            geo=dict(
                scope='world',
                projection=dict(type='mercator'),
                lonaxis=dict(range=[33, 42.5]),
                lataxis=dict(range=[-5, 5.5]),
                showframe=False,
                showcoastlines=False,
                bgcolor='rgba(0,0,0,0)'
            ),
            margin=dict(t=40, b=0, l=0, r=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Map figure generation failed: {str(e)}')


@app.get("/map/data")
async def map_data(metric: str = 'customers', segment: str = 'All'):
    """Return county-level aggregates for the selected metric and segment as JSON."""
    global last_results_df
    if last_results_df is None:
        raise HTTPException(status_code=400, detail='No results available. Upload data first.')

    try:
        df = last_results_df.copy()
        if segment != 'All':
            df = df[df['cluster_name'] == segment]
        if 'county' not in df.columns:
            raise HTTPException(status_code=400, detail='Uploaded data lacks county column for map aggregation.')

        if metric == 'customers':
            agg = df.groupby('county').size().rename('value').reset_index()
        elif metric == 'engagement':
            if 'engagement_rate' not in df.columns:
                raise HTTPException(status_code=400, detail='Uploaded data lacks engagement_rate for map aggregation.')
            agg = df.groupby('county')['engagement_rate'].mean().rename('value').reset_index()
        elif metric == 'share':
            total = len(last_results_df) if segment == 'All' else len(df)
            if total == 0:
                total = 1
            counts = df.groupby('county').size()
            agg = (counts / total * 100).rename('value').reset_index()
        else:
            raise HTTPException(status_code=400, detail='Unknown metric')

        data = agg.rename(columns={'county': 'County', 'value': 'Value'}).sort_values('Value', ascending=False).to_dict(orient='records')
        stats = {
            'counties': int(agg.shape[0]),
            'min': float(agg['value'].min()) if agg.shape[0] else 0.0,
            'max': float(agg['value'].max()) if agg.shape[0] else 0.0,
            'metric': metric,
            'segment': segment
        }
        return { 'data': data, 'stats': stats }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Map data aggregation failed: {str(e)}')

@app.get("/download-sample")
async def download_sample_data():
    """Download sample data file"""
    try:
        sample_file = "../data/sample_submission.csv"
        if os.path.exists(sample_file):
            return FileResponse(
                sample_file,
                media_type='text/csv',
                filename='sample_data.csv'
            )
        else:
            raise HTTPException(status_code=404, detail="Sample file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.get("/download-test")
async def download_test_data():
    """Download full test data (features only)"""
    try:
        test_file = "../data/test_data.csv"
        if os.path.exists(test_file):
            return FileResponse(
                test_file,
                media_type='text/csv',
                filename='test_data.csv'
            )
        else:
            raise HTTPException(status_code=404, detail="Test data file not found. Generate data first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.get("/download-train")
async def download_train_data():
    """Download full train data (with labels)"""
    try:
        train_file = "../data/train_data.csv"
        if os.path.exists(train_file):
            return FileResponse(
                train_file,
                media_type='text/csv',
                filename='train_data.csv'
            )
        else:
            raise HTTPException(status_code=404, detail="Train data file not found. Generate data first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
