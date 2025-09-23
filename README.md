### Map (Phase 2) Endpoints

These are scaffolded and ready but the UI map is deferred to Phase 2:

- `GET /map/figure` — returns a Plotly figure JSON of a Kenya county choropleth (customers/share/engagement). Uses the last uploaded dataset held in memory.
- `GET /map/data` — returns county aggregates with stats for legend/min–max and CSV download.

The frontend currently shows a "Kenya County Map (Coming Soon)" placeholder; re‑enabling the Map tab is part of Phase 2 (see Roadmap).

## 🖥️ Using the App (Frontend)

1. **Open** `http://localhost:3000`
2. **Upload Data**
   - Drag & drop a CSV or click "browse" on the Upload tab.
   - If you need a template, click "Download Full Test CSV". This includes all required columns.
3. **Results**
   - After upload, you’ll be redirected to the Results dashboard.
   - Use the tabs to explore: **Overview**, **Segment Details**, **Charts**.

## 📊 Results Tabs — What They Mean

- **Overview**
  - Shows Silhouette Score (how cleanly customers were grouped), Total Customers, Segments Found, and Target Status.
  - Tip: Start planning with the largest segments first; improve model features later.

- **Segment Details**
  - Each card shows: size and share, average age, engagement, purchases, Top Counties, Top Behaviors.
  - Under each card you’ll see a short **marketing insight** explaining who they are and how to act quickly.
  - Tip: Use **Top Counties** to choose regions and **Top Behaviors** to pick SKUs/promos.

- **Charts**
  - Segment Distribution bars (where the biggest wins are).
  - Average Age / Engagement by segment (to pick creative and channels).

## 🧭 Marketing Insights (Examples)

- **Budget Conscious** – Younger, highly engaged, respond to aspirational “affordable premium” offers.
  - Channel: Social + mobile offers
  - Offer: Affordable premium bundles, trial packs, time‑bound discounts
  - Message: “Upgrade tonight without breaking the bank.”

- **Urban Youth** – Frequent mainstream buyers; push convenience and rewards.
  - Channel: Loyalty, POS prompts, retail partners
  - Offer: Buy‑X‑get‑Y on Tusker/Guinness, instant rewards
  - Message: “Your go‑to brands, always in stock—earn as you buy.”

- **Premium Urban** – Selective and occasion‑driven; traditional + variety.
  - Channel: Event/holiday campaigns, SMS reminders
  - Offer: Occasion bundles (premium + local mix)
  - Message: “Celebrate moments with the right mix—classic and premium.”

- **Social Millennials** – Social, price‑sensitive; ideal for sampling/group offers.
  - Channel: Social + on‑prem activations
  - Offer: Group deals, happy‑hour promos, gamified coupons
  - Message: “Get together, try something new—save with friends.”

## 📥 Data Actions (Simplified)

- The Upload page is streamlined to keep one primary path:
  - **Download Full Test CSV** — a template with all required columns.
  - **Upload CSV** — run segmentation and view results.

## 🎨 UI & Theme

- Modern, v0‑style aesthetic: indigo/emerald palette, cards, buttons, and high‑contrast text.
- Hero is a subtle shiny grey gradient for a neutral, professional look.

# 🍺 EABL Insights Customer Segmentation Challenge

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-blue.svg)](https://reactjs.org)
[![Silhouette Score](https://img.shields.io/badge/Target%20Score-%3E0.7-gold.svg)](#model-performance)

A comprehensive customer segmentation challenge inspired by Kenyan market dynamics, featuring EABL's 2025 profit surge and Safaricom's M-Pesa upgrades. This project demonstrates advanced data science skills through unsupervised learning with a target Silhouette score >0.7.

## 🎯 Project Overview

This portfolio project simulates a Kaggle-like hackathon challenge focused on customer segmentation for marketing and CRM applications in the Kenyan market. It incorporates real-world business context from:

- **EABL's FY2025 Performance**: 12% profit surge to KSh 12.2 billion, driven by premium spirits
- **Safaricom's M-Pesa 2025 Upgrade**: Enhanced capacity to 6,000 TPS with cloud-native architecture

### Key Features

- 🔢 **Synthetic Dataset**: 80,000+ customer records with Kenyan market context
- 🤖 **Advanced ML**: K-means clustering with PCA optimization for Silhouette score >0.7
- 🌐 **Full-Stack App**: FastAPI backend + React frontend with modern UI
- 📊 **Interactive Dashboards**: Real-time visualizations and segment analysis
- 🏆 **Evaluation System**: Automated scoring and leaderboard simulation

## 📁 Project Structure

```
EABLInsights/
├── 📊 data/                     # Generated datasets
│   ├── train_data.csv          # Training data (80% with labels)
│   ├── test_data.csv           # Test data (20% without labels)
│   ├── test_labels.csv         # Ground truth for evaluation
│   └── sample_submission.csv   # Submission format template
├── 🔬 notebooks/               # Analysis notebooks
│   └── customer_segmentation_analysis.ipynb
├── 🚀 backend/                 # FastAPI application
│   └── main.py                 # API endpoints and ML inference
├── ⚛️ frontend/                # Next.js + Tailwind React application
├── 📜 scripts/                 # Utility scripts
│   ├── generate_synthetic_data.py
│   └── evaluation.py
├── 🤖 models/                  # Trained model artifacts
├── 📚 docs/                    # Documentation
└── 📋 requirements.txt         # Python dependencies
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone or navigate to project directory
cd "C:\Users\user\OneDrive\Desktop\Hackathon 2025\EABLInsights"

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Data

```bash
# Generate 80,000 customer records
python scripts/generate_synthetic_data.py
```

This creates:
- **Training data**: 64,000 records with segment labels
- **Test data**: 16,000 records without labels
- **Kenyan context**: Counties, income brackets, behavior patterns
- **2025 features**: M-Pesa upgrade engagement, profit segments

### 3. Run Analysis Notebook

```bash
# Start Jupyter
jupyter notebook notebooks/customer_segmentation_analysis.ipynb
```

The notebook will:
- ✅ Perform comprehensive EDA
- ✅ Optimize K-means clustering with PCA
- ✅ Achieve Silhouette score >0.7
- ✅ Save trained model components

### 4. Start Backend API

```bash
# Run FastAPI server
cd backend
python main.py
```

API will be available at `http://localhost:8000` with endpoints:
- `POST /upload` - Upload CSV for segmentation
- `GET /generate-data` - Generate sample data
- `POST /visualize` - Create cluster visualizations
- `GET /model-info` - Model performance metrics

### 5. Start Frontend (Next.js)

```bash
# In a new terminal
cd frontend
npm install
npm run dev
```

Frontend will be available at `http://localhost:3000`.

### 6. Test the System

```bash
# Evaluate sample submission
python scripts/evaluation.py
```

## 🎯 Model Performance

Our clustering model achieves the target performance:

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Silhouette Score | >0.7 | 0.7142 | ✅ **PASSED** |
| Clusters | 12 | 12 | ✅ Current |
| Processing Time | <2s | ~1.5s | ✅ Fast |
| Model Reload | Auto-detection | ✅ Enabled | ✅ Hot-reload |

### Technical Approach

1. **Feature Engineering**: Encode categorical variables, standardize numerical features
2. **Dimensionality Reduction**: PCA with 4-6 components (90%+ variance explained)
3. **Clustering Optimization**: Grid search for optimal k with silhouette scoring
4. **Model Pipeline**: Scikit-learn pipeline for reproducible preprocessing

### 🔧 Recent Improvements

- **Auto-Reload**: Backend automatically detects model changes and reloads without restart
- **Unified Display**: Frontend consistently shows model performance (0.7142) across all metrics
- **Target Update**: Raised target silhouette score from 0.6 to 0.7 for higher standards
- **Training Optimization**: Added `--no-viz` flag for faster training without visualizations
- **Score Consistency**: Fixed discrepancies between dataset and model silhouette scores

## 📊 Dataset Schema

| Column | Description | Type | Example |
|--------|-------------|------|---------|
| `customer_id` | Unique identifier | Integer | 3344 |
| `age` | Customer age | Integer | 30 |
| `income_kes` | Income bracket | String | "50k-100k" |
| `purchase_history` | Transaction count | Integer | 15 |
| `behavior` | Product preferences | String | "Tusker,Vooma" |
| `engagement_rate` | Marketing engagement | Float | 0.45 |
| `county` | Kenyan county | String | "Nairobi" |
| `profit_segment` | EABL profit category | String | "High-Spirits" |
| `upgrade_engagement` | M-Pesa 2025 boost | Float | 1.2 |
| `segment_target` | Ground truth label | String | "Urban Youth" |

## 🎨 Customer Segments

This demo currently exposes 12 customer segments end‑to‑end in the UI (auto‑labeled from Top Behaviors):

- Premium Spirits
- Mainstream Loyalists (Tusker/Guinness)
- Budget Conscious
- Mixed Portfolio
- Local Brews
- Premium Consumer
- Traditionalists (White Cap / Pilsner)
- Urban Youth
- Social Millennials
- Weekend Warriors
- Occasional Drinker
- Corporate Professional

## 🌍 Kenyan Market Context

### EABL 2025 Performance Integration
- **Profit Segments**: Aligned with spirits (KSh 5.2B) and beer performance
- **Premium Focus**: Johnnie Walker, Kenya Cane success reflected in segments
- **Market Challenges**: Illicit alcohol competition considered in pricing segments

### Safaricom M-Pesa 2025 Upgrade Features
- **Upgrade Engagement**: Measures customer adoption of new features
- **Transaction Capacity**: 6,000 TPS baseline, 12,000 TPS scalable
- **Cloud-Native**: Faster deployment cycles reflected in engagement metrics

## 🛠️ API Documentation

### Upload Endpoint
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_data.csv"
```

Response:
```json
{
  "status": "success",
  "silhouette_score": 0.8734,
  "model_performance": "Excellent",
  "total_customers": 16000,
  "n_clusters": 10,
  "cluster_summary": [...],
  "predictions": [...]
}
```

### Visualization Endpoint
```bash
curl -X POST "http://localhost:8000/visualize" \
  -H "Content-Type: application/json" \
  -d '{"predictions": [...]}'
```

## 🏆 Evaluation Metrics

The evaluation system provides comprehensive scoring:

```python
from scripts.evaluation import SegmentationEvaluator

evaluator = SegmentationEvaluator()
metrics = evaluator.evaluate_submission("my_submission.csv")

print(f"Silhouette Score: {metrics['silhouette_score']}")
print(f"Target Achieved: {metrics['target_achieved']}")
```

### Leaderboard Simulation
```python
from scripts.evaluation import LeaderboardSimulator

leaderboard = LeaderboardSimulator()
leaderboard.add_submission("My Team", metrics)
leaderboard.display_leaderboard()
```

## 🎯 Challenge Rules

1. **Objective**: Achieve Silhouette score >0.85 on test data
2. **Data**: Use provided synthetic dataset only
3. **Submission**: CSV with `customer_id` and `segment_target` columns
4. **Evaluation**: Automated scoring with clustering metrics
5. **Leaderboard**: Ranked by Silhouette score, then by submission time

## 🚀 Deployment

### Local Development
```bash
# Backend
cd backend && python main.py

# Frontend (when created)
cd frontend && npm start
```

### Production Deployment
- **Backend**: Deploy FastAPI to Heroku/Railway
- **Frontend**: Deploy React to Vercel/Netlify
- **Database**: SQLite for demo, PostgreSQL for production

## 📈 Performance Optimization

### Achieving Silhouette Score >0.85

1. **Feature Selection**: Focus on discriminative features
2. **Scaling**: StandardScaler for numerical features
3. **PCA**: Reduce dimensionality while preserving variance
4. **Hyperparameter Tuning**: Grid search for optimal k
5. **Ensemble Methods**: Consider multiple clustering algorithms

### Code Optimization
- Vectorized operations with NumPy/Pandas
- Efficient data structures
- Caching for repeated computations
- Async API endpoints for scalability

## 🤝 Contributing

This is a portfolio project, but suggestions are welcome:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is for educational and portfolio purposes. The synthetic data and business context are inspired by real companies but are not affiliated with or endorsed by EABL or Safaricom.

## 🙏 Acknowledgments

- **East African Breweries Limited**: Market context and business insights
- **Safaricom**: M-Pesa innovation and Kenyan fintech leadership
- **Kenyan Market**: Rich consumer behavior patterns and demographics
- **Open Source Community**: Libraries and tools that made this possible

## 📞 Contact

**Portfolio Project by**: [Your Name]
**GitHub**: [Your GitHub Profile]
**LinkedIn**: [Your LinkedIn Profile]
**Email**: [Your Email]

---

**Disclaimer**: This is a simulated challenge for portfolio demonstration. All data is synthetic and generated for educational purposes. Not affiliated with EABL, Safaricom, or any real competition.

---

## 🗺️ Roadmap — Phase 2 (Map & Geo Insights)

The Results page currently shows a "Kenya County Map (Coming Soon)" placeholder. In Phase 2 we will ship a full, interactive map experience with:

- **Choropleth Map (Kenya Only)**
  - Customers per county (default)
  - Segment share (%) per county
  - Average engagement per county
- **Legend & Stats**
  - Colorbar, min/max, county count, active metric/segment
- **Download Map Data (CSV)**
  - County-level aggregates for the selected metric and segment
- **Backend Aggregation**
  - Exact county totals computed during inference
  - Server-generated Plotly figure returned from `GET /map/figure`
- **Offline Reliability**
  - Local Kenya GeoJSON bundled for no-network environments

These features are already scaffolded in the codebase:

- Backend endpoints in `backend/main.py`:
  - `GET /map/figure` (Plotly figure JSON)
  - `GET /map/data` (county aggregates + stats)
- Frontend wiring in `frontend/src/components/Dashboard.tsx` and `ServerMap.tsx` is ready to be re-enabled once the map is unhidden.

If you’d like to pull this forward, open an issue or ping me and I’ll enable the map tab and legend in the UI.

---

## 🧪 Score Improvement Plan (Feature Engineering & Algorithms)

Below is a practical roadmap to improve the Silhouette score and segment separation:

- **Feature Engineering**
  - Encode recency/frequency/monetary‑like signals from `purchase_history` (e.g., log‑scaled, capped, or z‑score).
  - Interaction features: `age × engagement_rate`, `upgrade_engagement × income_kes` (ordinalized), `profit_segment` one‑hots.
  - Geography features: one‑hot or target‑encoding for `county`; regional roll‑ups (e.g., Nairobi metro vs. coast vs. west).
  - Behavioral embeddings: break `behavior` strings into categories (e.g., premium, local, mainstream) and one‑hot/weights.

- **Preprocessing**
  - Scale numerics with `StandardScaler`; try `RobustScaler` if heavy tails.
  - Reduce dimensionality with PCA (3–6 components); tune components to maximize Silhouette.

- **Clustering Algorithms**
  - K‑means (current): grid‑search k and PCA components.
  - Gaussian Mixture Models (GMM) to capture elliptical clusters.
  - HDBSCAN/DBSCAN for density‑based segmentation if clusters are non‑spherical.
  - MiniBatchKMeans for speed on large data.

- **Model Selection & Validation**
  - Use a Silhouette‑guided search over (k, PCA components, scaler).
  - Add Calinski–Harabasz and Davies–Bouldin indices for tie‑breaking.
  - Bootstrapped Silhouette to test stability of segments.

- **Operational Suggestions**
  - Cap outliers (e.g., top 1%) for purchase/engagement to reduce cluster bleed.
  - Balance segments by reweighing features that dominate (e.g., income buckets).
  - Revisit `behavior` taxonomy for clearer separation (e.g., premium/spirit/mainstream/local).

These steps can be integrated into the notebook and training script (`scripts/train_model.py`). Once a better configuration is found, re‑save PCA/KMeans and redeploy.
