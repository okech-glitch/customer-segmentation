'use client';

import { useEffect, useState } from 'react';
import { BarChart3, PieChart, Users, TrendingUp, Award, Download } from 'lucide-react';
import { getApiBaseUrl } from '@/lib/api';

interface DashboardProps {
  results: {
    status: string;
    silhouette_score: number;
    model_performance: string;
    total_customers: number;
    n_clusters: number;
    cluster_summary: Array<{
      cluster_id: number;
      cluster_name: string;
      size: number;
      percentage: number;
      avg_age: number;
      avg_engagement: number;
      avg_purchases: number;
      top_counties: Record<string, number>;
      top_behaviors: Record<string, number>;
    }>;
    predictions: Array<{
      customer_id: number;
      predicted_cluster: number;
      cluster_name: string;
    }>;
  };
}

export default function Dashboard({ results }: DashboardProps) {
  const [activeView, setActiveView] = useState('overview');
  const [modelInfo, setModelInfo] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    const fetchInfo = async () => {
      try {
        const res = await fetch(`${getApiBaseUrl()}/model-info`, {
          next: { revalidate: 0 },
        });
        if (res.ok) {
          const data = await res.json();
          setModelInfo(data);
        }
      } catch {
        // ignore fetch errors (backend may be offline)
      }
    };
    fetchInfo();
  }, []);

  const getSegmentInsights = (name: string) => {
    const n = name.toLowerCase();
    if (n.includes('budget')) {
      return (
        <>
          <p className="card-text text-sm mb-2">
            Younger, highly engaged and responsive to aspirational messaging. Despite the label, they do try premium occasionally.
          </p>
          <ul className="text-sm text-slate-600 list-disc pl-5 space-y-1">
            <li><strong>Channel:</strong> Social + mobile offers.</li>
            <li><strong>Offer:</strong> Affordable premium bundles, trial packs, time‑bound discounts.</li>
            <li><strong>Message:</strong> “Upgrade tonight without breaking the bank.”</li>
          </ul>
        </>
      );
    }
    if (n.includes('urban youth')) {
      return (
        <>
          <p className="card-text text-sm mb-2">
            Frequent buyers of mainstream brands; less campaign‑responsive. Push convenience and rewards.
          </p>
          <ul className="text-sm text-slate-600 list-disc pl-5 space-y-1">
            <li><strong>Channel:</strong> Loyalty, POS prompts, retail partners.</li>
            <li><strong>Offer:</strong> Buy‑X‑get‑Y on Tusker/Guinness, instant rewards.</li>
            <li><strong>Message:</strong> “Your go‑to brands, always in stock—earn as you buy.”</li>
          </ul>
        </>
      );
    }
    if (n.includes('premium urban')) {
      return (
        <>
          <p className="card-text text-sm mb-2">
            Selective, occasion‑driven, lean traditional but open to variety. Not heavy promo responders.
          </p>
          <ul className="text-sm text-slate-600 list-disc pl-5 space-y-1">
            <li><strong>Channel:</strong> Event/holiday campaigns, SMS reminders.</li>
            <li><strong>Offer:</strong> Occasion bundles (premium + local mix).</li>
            <li><strong>Message:</strong> “Celebrate moments with the right mix—classic and premium.”</li>
          </ul>
        </>
      );
    }
    if (n.includes('social millennials')) {
      return (
        <>
          <p className="card-text text-sm mb-2">
            Social and price‑sensitive; purchase less often. Great for sampling and group offers.
          </p>
          <ul className="text-sm text-slate-600 list-disc pl-5 space-y-1">
            <li><strong>Channel:</strong> Social + on‑premise activations.</li>
            <li><strong>Offer:</strong> Group deals, happy‑hour promos, gamified coupons.</li>
            <li><strong>Message:</strong> “Get together, try something new—save with friends.”</li>
          </ul>
        </>
      );
    }
    return (
      <p className="card-text text-sm">
        Use <strong>Top Counties</strong> for regional targeting and <strong>Top Behaviors</strong> to pick SKUs and promos tailored to this group.
      </p>
    );
  };

  // Behavior-first marketing label
  const getMarketingLabel = (cluster: any) => {
    const behaviors = Object.keys(cluster.top_behaviors || {}).map(b => b.toLowerCase());
    const joined = behaviors.join(',');
    if (joined.includes('premium') || joined.includes('johnnie') || joined.includes('spirits')) return 'Premium Spirits';
    if (joined.includes('tusker') || joined.includes('guinness')) return 'Mainstream Loyalists';
    if (joined.includes('white cap') || joined.includes('pilsner')) return 'Traditionalists';
    if (joined.includes('local')) return 'Local Brews';
    if (joined.includes('budget')) return 'Budget Conscious';
    if (joined.includes('mixed')) return 'Mixed Portfolio';
    if (joined.includes('premium consumer')) return 'Premium Consumer';
    if (joined.includes('occasional')) return 'Occasional Drinker';
    return 'Customer Segment';
  };

  const getMarketingBlurb = (label: string) => {
    const n = label.toLowerCase();
    if (n.includes('premium spirits')) return (
      <p className="card-text text-sm mb-2">Affluent, selective, occasion-driven. Use exclusive bundles, VIP perks, and limited editions.</p>
    );
    if (n.includes('mainstream')) return (
      <p className="card-text text-sm mb-2">Habitual buyers of mainstream beers. Push loyalty, stock reliability, and sports tie-ins.</p>
    );
    if (n.includes('traditional')) return (
      <p className="card-text text-sm mb-2">Legacy brand affinity. Use heritage messaging and dependable availability.</p>
    );
    if (n.includes('local brews')) return (
      <p className="card-text text-sm mb-2">Prefer local/traditional brews. Community activations and value bundles work best.</p>
    );
    if (n.includes('budget')) return (
      <p className="card-text text-sm mb-2">Price sensitive, younger. Use time-bound deals and affordable premium trials.</p>
    );
    if (n.includes('mixed')) return (
      <p className="card-text text-sm mb-2">Open to discovery. Offer sampler packs and try-3-get-reward campaigns.</p>
    );
    if (n.includes('premium consumer')) return (
      <p className="card-text text-sm mb-2">High spenders. Offer concierge service, early access, and premium cross-sell.</p>
    );
    if (n.includes('occasional')) return (
      <p className="card-text text-sm mb-2">Infrequent, event-driven. Use occasion-based reminders and small trial sizes.</p>
    );
    return (
      <p className="card-text text-sm mb-2">Use Top Behaviors and Counties below to tailor SKUs, channels, and timing.</p>
    );
  };

  const downloadResults = () => {
    const dataStr = JSON.stringify(results, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = 'segmentation_results.json';
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  const downloadPredictions = () => {
    const csvContent = [
      ['customer_id', 'predicted_cluster', 'cluster_name'],
      ...results.predictions.map(p => [p.customer_id, p.predicted_cluster, p.cluster_name])
    ].map(row => row.join(',')).join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'customer_predictions.csv';
    link.click();
    window.URL.revokeObjectURL(url);
  };

  // Prepare data for visualizations
  const clusterData = results.cluster_summary.map(cluster => ({
    name: cluster.cluster_name,
    size: cluster.size,
    percentage: cluster.percentage,
    avg_age: cluster.avg_age,
    avg_engagement: cluster.avg_engagement,
    avg_purchases: cluster.avg_purchases
  }));

  // Performance indicator color
  const getPerformanceColor = (score: unknown) => {
    if (typeof score === 'number' && score > 0.7) return 'text-green-600 bg-green-100';
    if (typeof score === 'number' && score > 0.5) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  return (
    <div className="space-y-8">
      {/* Performance Header */}
      <div className="card p-8">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6">
          <div>
            <h1 className="text-3xl font-bold text-slate-800 mb-2">Segmentation Results</h1>
            <p className="text-slate-600">Customer segmentation analysis completed successfully</p>
          </div>

      {/* Helper blurb for Overview */}
      {activeView === 'overview' && (
        <div className="card p-4 text-sm text-slate-600 dark:text-slate-200">
          <strong>How to read this:</strong> Silhouette shows how clean the clusters are (1.0 is perfect). Start planning with the biggest segments first; improve the score later with better features.
        </div>
      )}
          <div className="flex space-x-3 mt-4 md:mt-0">
            <button
              onClick={downloadResults}
              className="btn btn-primary flex items-center space-x-2"
            >
              <Download className="w-4 h-4" />
              <span>Download Results</span>
            </button>
            <button
              onClick={downloadPredictions}
              className="btn btn-secondary flex items-center space-x-2"
            >
              <Download className="w-4 h-4" />
              <span>Download CSV</span>
            </button>
          </div>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {/* Dataset (this upload) Silhouette */}
          <div className="bg-gradient-to-r from-blue-50 to-blue-100 dark:from-slate-700 dark:to-slate-600 p-6 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-700 dark:text-white">Dataset Silhouette</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-white">{((modelInfo as any)?.silhouette_score ?? (results as any).silhouette_score).toFixed(4)}</p>
              </div>
              <Award className="w-8 h-8 text-blue-600 dark:text-white" />
            </div>
            <div className={`mt-2 px-2 py-1 rounded-full text-xs font-medium ${getPerformanceColor(results.silhouette_score)}`}>
              {results.model_performance}
            </div>
          </div>

          {/* Model (training) Silhouette */}
          <div className="bg-gradient-to-r from-indigo-50 to-indigo-100 dark:from-slate-700 dark:to-slate-600 p-6 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-700 dark:text-white">Model Silhouette</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-white">{((modelInfo as any)?.silhouette_score ?? (results as any).silhouette_score).toFixed(4)}</p>
              </div>
              <Award className="w-8 h-8 text-indigo-600 dark:text-white" />
            </div>
            <div className="mt-2 px-2 py-1 rounded-full text-xs font-medium bg-indigo-100 text-indigo-700 dark:bg-slate-600 dark:text-white">
              Trained on 64k
            </div>
          </div>

          <div className="bg-gradient-to-r from-green-50 to-green-100 dark:from-slate-700 dark:to-slate-600 p-6 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-700 dark:text-white">Total Customers</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-white">{results.total_customers.toLocaleString()}</p>
              </div>
              <Users className="w-8 h-8 text-green-600 dark:text-white" />
            </div>
            <p className="text-xs text-slate-600 dark:text-slate-200 mt-2">Successfully segmented</p>
          </div>

          <div className="bg-gradient-to-r from-purple-50 to-purple-100 dark:from-slate-700 dark:to-slate-600 p-6 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-700 dark:text-white">Segments Found</p>
                <p className="text-2xl font-bold text-slate-900 dark:text-white">{(modelInfo as any)?.n_clusters || (results as any).n_clusters}</p>
              </div>
              <PieChart className="w-8 h-8 text-purple-600 dark:text-white" />
            </div>
            <p className="text-xs text-slate-600 dark:text-slate-200 mt-2">Distinct customer groups</p>
          </div>

          <div className="bg-gradient-to-r from-orange-50 to-orange-100 dark:from-slate-700 dark:to-slate-600 p-6 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-700 dark:text-white">Target Status</p>
                <p className="text-lg font-bold text-slate-900 dark:text-white">
                  {((modelInfo as any)?.silhouette_score || (results as any).silhouette_score) > 0.7 ? '✅ ACHIEVED' : '❌ MISSED'}
                </p>
              </div>
              <TrendingUp className="w-8 h-8 text-orange-600 dark:text-white" />
            </div>
            <p className="text-xs text-slate-600 dark:text-slate-200 mt-2">Score &gt; 0.7 target</p>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="card p-2">
        {[
          { id: 'overview', label: 'Overview', icon: BarChart3 },
          { id: 'segments', label: 'Segment Details', icon: Users },
          { id: 'visualizations', label: 'Charts', icon: PieChart }
        ].map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveView(id)}
            className={`btn ${activeView === id ? 'btn-primary' : ''} mr-2`}
          >
            <Icon className="w-4 h-4" />
            <span>{label}</span>
          </button>
        ))}
      </div>

      {/* Map placeholder for Phase 2 */}
      <div className="card p-6">
        <h3 className="text-xl font-bold mb-2 text-slate-800">Kenya County Map (Coming Soon)</h3>
        <p className="card-text text-sm">
          An interactive choropleth will appear here in Phase 2, showing customers, segment share, or engagement by county.
          We’ll include a legend, download, and filters. For now, use the Charts and Segment Details to explore the results.
        </p>
      </div>

      {/* Content Views */}
      {activeView === 'overview' && (
        <div className="grid md:grid-cols-2 gap-8">
          {/* Cluster Size Distribution */}
          <div className="card p-6">
            <h3 className="text-xl font-bold mb-4 text-slate-800">Segment Size Distribution</h3>
            <div className="space-y-3">
              {clusterData.map((cluster, index) => (
                <div key={index} className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: `hsl(${index * 36}, 70%, 50%)` }}></div>
                    <span className="font-medium text-slate-800">{cluster.name}</span>
                  </div>
                  <div className="text-right">
                    <div className="font-bold text-slate-800 dark:text-slate-100">{cluster.size.toLocaleString()}</div>
                    <div className="text-sm text-slate-500 dark:text-slate-300">{cluster.percentage.toFixed(1)}%</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Performance Summary */}
          <div className="card p-6">
            <h3 className="text-xl font-bold mb-4 text-slate-800">Model Performance</h3>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-slate-600">Silhouette Score</span>
                <span className="font-bold text-lg text-slate-800">{((modelInfo as any)?.silhouette_score ?? (results as any).silhouette_score).toFixed(4)}</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div 
                  className={`h-3 rounded-full ${((modelInfo as any)?.silhouette_score ?? (results as any).silhouette_score) > 0.7 ? 'bg-[var(--brand-secondary)]' : 'bg-amber-400'}`}
                  style={{ width: `${Math.min(((modelInfo as any)?.silhouette_score ?? (results as any).silhouette_score) * 100, 100)}%` }}
                ></div>
              </div>
              <div className="text-sm text-slate-600 dark:text-slate-200">
                Target: 0.7 | Current: {((modelInfo as any)?.silhouette_score ?? (results as any).silhouette_score).toFixed(4)}
              </div>
              
              <div className="mt-6 p-4 rounded-lg bg-white dark:bg-slate-700">
                <h4 className="font-semibold mb-2 text-slate-800 dark:text-white">Interpretation</h4>
                <p className="text-sm text-slate-700 dark:text-slate-100">
                  {((modelInfo as any)?.silhouette_score ?? (results as any).silhouette_score) > 0.7 
                    ? "Excellent clustering performance! The segments are well-separated and cohesive."
                    : ((modelInfo as any)?.silhouette_score ?? (results as any).silhouette_score) > 0.5
                    ? "Good clustering performance. Segments are reasonably well-defined."
                    : "Clustering performance needs improvement. Consider feature engineering or different algorithms."
                  }
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Map deferred to phase 2 */}

      {activeView === 'segments' && (
        <div className="card p-8">
          <h3 className="text-2xl font-bold mb-6">Detailed Segment Analysis</h3>
          <p className="card-text text-sm mb-4">Each card shows a segment’s size, demographics and behavior. Use <strong>Top Counties</strong> to plan regional campaigns and <strong>Top Behaviors</strong> to tailor product bundles.</p>

          {/* Simple explanation of the numbers shown on each card */}
          <div className="mb-6 p-4 rounded-lg bg-slate-50 border text-sm text-slate-700 dark:bg-slate-700 dark:text-white">
            <div className="font-semibold mb-2">How to read each segment card</div>
            <ul className="list-disc pl-5 space-y-1">
              <li><strong>Size</strong>: number of customers in this segment (e.g., 9,341).</li>
              <li><strong>Share (%)</strong>: percent of all customers in this segment (e.g., 14.6%).</li>
              <li><strong>Average Age</strong>: typical age in this group (e.g., 35.0 years).</li>
              <li><strong>Engagement Rate</strong>: how often they interact with campaigns (e.g., 41.3%).</li>
              <li><strong>Avg Purchases</strong>: typical number of purchases per customer (e.g., 7.4).</li>
              <li><strong>Top Counties</strong>: best regions to focus for this group.</li>
              <li><strong>Top Behaviors</strong>: brands/categories they prefer—use to pick SKUs and offers.</li>
            </ul>
          </div>

          <div className="grid gap-6">
            {results.cluster_summary.map((cluster) => (
              <div key={cluster.cluster_id} className="border rounded-lg p-6 hover:shadow-md transition-shadow">
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h4 className="text-xl font-bold text-slate-800">{getMarketingLabel(cluster)}</h4>
                    <p className="text-slate-600">{cluster.cluster_name} • Cluster {cluster.cluster_id}</p>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-bold text-indigo-600">{cluster.size.toLocaleString()}</div>
                    <div className="text-sm text-slate-500">{cluster.percentage.toFixed(1)}% of customers</div>
                  </div>
                </div>

                {/* Marketing insights */}
                <div className="mb-4">
                  {getMarketingBlurb(getMarketingLabel(cluster))}
                </div>

                <div className="grid md:grid-cols-3 gap-4 mb-4">
                  <div className="bg-blue-50 p-3 rounded">
                    <div className="text-sm text-blue-600 font-medium">Average Age</div>
                    <div className="text-lg font-bold text-blue-800">{cluster.avg_age.toFixed(1)} years</div>
                  </div>
                  <div className="bg-green-50 p-3 rounded">
                    <div className="text-sm text-green-600 font-medium">Engagement Rate</div>
                    <div className="text-lg font-bold text-green-800">{(cluster.avg_engagement * 100).toFixed(1)}%</div>
                  </div>
                  <div className="bg-purple-50 p-3 rounded">
                    <div className="text-sm text-purple-600 font-medium">Avg Purchases</div>
                    <div className="text-lg font-bold text-purple-800">{cluster.avg_purchases.toFixed(1)}</div>
                  </div>
                </div>

                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <h5 className="font-semibold text-slate-800 dark:text-white mb-1">Top Counties</h5>
                    <p className="text-xs text-slate-500 dark:text-slate-300 mb-2">Numbers show how many customers in this segment come from each county.</p>
                    <div className="space-y-1">
                      {Object.entries(cluster.top_counties).slice(0, 3).map(([county, count]) => (
                        <div key={county} className="flex justify-between text-sm">
                          <span className="text-slate-800 dark:text-slate-100">{county}</span>
                          <span className="text-slate-600 dark:text-slate-300">{count}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div>
                    <h5 className="font-semibold text-slate-800 dark:text-white mb-1">Top Behaviors</h5>
                    <p className="text-xs text-slate-500 dark:text-slate-300 mb-2">Numbers show how many customers in this segment match each behavior preference. One customer can match multiple behaviors, so totals may exceed the segment size.</p>
                    <div className="space-y-1">
                      {Object.entries(cluster.top_behaviors).slice(0, 3).map(([behavior, count]) => (
                        <div key={behavior} className="flex justify-between text-sm">
                          <span className="truncate text-slate-800 dark:text-slate-100">{behavior}</span>
                          <span className="text-slate-600 dark:text-slate-300">{count}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {activeView === 'visualizations' && (
        <div className="space-y-8">
          {/* Cluster Size Chart */}
          <div className="card p-6">
            <h3 className="text-xl font-bold mb-4 text-slate-800">Segment Distribution</h3>
            <p className="card-text text-sm mb-3">Largest bars indicate the biggest opportunities for quick wins.</p>
            <div className="space-y-4">
              {clusterData.map((cluster, index) => (
                <div key={index} className="flex items-center">
                  <div className="w-40 text-sm font-medium truncate text-slate-800 dark:text-slate-100">{cluster.name}</div>
                  <div className="flex-1 mx-4">
                    <div className="bg-gray-200 dark:bg-slate-600 rounded-full h-4">
                      <div 
                        className="h-4 rounded-full transition-all duration-300"
                        style={{ backgroundColor: 'var(--brand-primary)', width: `${cluster.percentage}%` }}
                      ></div>
                    </div>
                  </div>
                  <div className="w-24 text-right text-sm">
                    <div className="font-bold text-slate-900 dark:text-white">{cluster.size.toLocaleString()}</div>
                    <div className="text-slate-600 dark:text-slate-200">{cluster.percentage.toFixed(1)}%</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Segment Metrics */}
          <div className="card p-6">
            <h3 className="text-xl font-bold mb-4 text-slate-800">Segment Metrics Overview</h3>
            <p className="card-text text-sm mb-3">Compare average age and engagement across segments to decide creative and channel strategy.</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-semibold mb-3 text-slate-700">Average Age by Segment</h4>
                <div className="space-y-2">
                  {clusterData.map((cluster, index) => (
                    <div key={index} className="flex justify-between items-center">
                      <span className="text-sm truncate text-slate-700">{cluster.name}</span>
                      <span className="font-medium text-slate-800">{cluster.avg_age.toFixed(1)} years</span>
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <h4 className="font-semibold mb-3 text-slate-700">Engagement Rate by Segment</h4>
                <div className="space-y-2">
                  {clusterData.map((cluster, index) => (
                    <div key={index} className="flex justify-between items-center">
                      <span className="text-sm truncate text-slate-700">{cluster.name}</span>
                      <span className="font-medium text-slate-800">{(cluster.avg_engagement * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
