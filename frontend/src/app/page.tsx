'use client';

import { useEffect, useState } from 'react';
import { Upload, BarChart3, Users, Target, Award, TrendingUp } from 'lucide-react';
import FileUpload from '@/components/FileUpload';
import Dashboard from '@/components/Dashboard';
import Header from '@/components/Header';
import { getApiBaseUrl } from '@/lib/api';

export default function Home() {
  const [activeTab, setActiveTab] = useState('upload');
  const [segmentationResults, setSegmentationResults] = useState<Record<string, unknown> | null>(null);
  const [modelInfo, setModelInfo] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    // Fetch current model info from backend to show live segments/silhouette
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
        // ignore if backend not running
      }
    };
    fetchInfo();
  }, []);

  const handleSegmentationComplete = (results: any) => {
    setSegmentationResults(results);
    setActiveTab('results');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">
      <Header />
      
      {/* Hero Section */}
      <div className="hero-section on-dark hero-shiny text-white py-16">
        <div className="container mx-auto px-4 text-center">
          <h1 className="text-5xl font-bold mb-4">🍺 EABL Insights Challenge</h1>
          <p className="text-xl mb-8 max-w-3xl mx-auto">
            Customer Segmentation for Kenyan Market
          </p>
          
          {/* Key Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mt-12">
            <div className="hero-tile card-hover p-6">
              <Users className="w-8 h-8 mb-2 mx-auto" />
              <div className="text-2xl font-bold">80K+</div>
              <div className="text-sm opacity-90">Customer Records</div>
            </div>
            <div className="hero-tile card-hover p-6">
              <Target className="w-8 h-8 mb-2 mx-auto" />
              <div className="text-2xl font-bold">&gt;0.7</div>
              <div className="text-sm opacity-90">
                {modelInfo?.silhouette_score ? (
                  <>Achieved: {Number(modelInfo.silhouette_score).toFixed(3)}</>
                ) : (
                  <>Target Score</>
                )}
              </div>
            </div>
            <div className="hero-tile card-hover p-6">
              <BarChart3 className="w-8 h-8 mb-2 mx-auto" />
              <div className="text-2xl font-bold">{(modelInfo as any)?.n_clusters ?? 12}</div>
              <div className="text-sm opacity-90">Customer Segments</div>
            </div>
            <div className="hero-tile card-hover p-6">
              <Award className="w-8 h-8 mb-2 mx-auto" />
              <div className="text-2xl font-bold">2025</div>
              <div className="text-sm opacity-90">Market Context</div>
            </div>
          </div>
        </div>
        <div className="hero-glow" />
      </div>

      {/* Navigation Tabs */}
      <div className="container mx-auto px-4 py-8">
        <div className="flex flex-wrap justify-center gap-3 mb-8">
          <button
            onClick={() => setActiveTab('upload')}
            className={`btn ${activeTab === 'upload' ? 'btn-primary' : 'btn-default'}`}
          >
            <Upload className="w-5 h-5 inline mr-2" />
            Upload Data
          </button>
          <button
            onClick={() => setActiveTab('results')}
            className={`btn ${activeTab === 'results' ? 'btn-primary' : 'btn-default'}`}
            disabled={!segmentationResults}
          >
            <TrendingUp className="w-5 h-5 inline mr-2" />
            Results Dashboard
          </button>
          <button
            onClick={() => setActiveTab('about')}
            className={`btn ${activeTab === 'about' ? 'btn-primary' : 'btn-default'}`}
          >
            <BarChart3 className="w-5 h-5 inline mr-2" />
            About Challenge
          </button>
        </div>

        {/* Tab Content */}
        <div className="max-w-6xl mx-auto">
          {activeTab === 'upload' && (
            <FileUpload onSegmentationComplete={handleSegmentationComplete} />
          )}
          
          {activeTab === 'results' && segmentationResults && (
            <Dashboard results={segmentationResults as any} />
          )}
          
          {activeTab === 'about' && (
            <div className="card p-8">
              <h2 className="text-3xl font-bold mb-6 text-gray-800">About the Challenge</h2>
              
              <div className="grid md:grid-cols-2 gap-8">
                <div>
                  <h3 className="text-xl font-semibold mb-4 text-blue-600">🎯 Objective</h3>
                  <p className="text-black mb-4">
                    Segment customers using unsupervised learning to achieve a Silhouette score greater than 0.7.
                    This challenge simulates real-world customer segmentation for Kenyan market leaders.
                  </p>
                  
                  <h3 className="text-xl font-semibold mb-4 text-green-600">🌍 Market Context</h3>
                  <ul className="text-black space-y-2">
                    <li>• EABL&apos;s 2025 profit surge: KSh 12.2B (+12%)</li>
                    <li>• Safaricom M-Pesa upgrade: 6,000 TPS capacity</li>
                    <li>• Premium spirits driving growth</li>
                    <li>• Cloud-native fintech architecture</li>
                  </ul>
                </div>
                
                <div>
                  <h3 className="text-xl font-semibold mb-4 text-purple-600">🔬 Technical Stack</h3>
                  <ul className="text-black space-y-2">
                    <li>• K-means clustering with PCA optimization</li>
                    <li>• Scikit-learn for machine learning</li>
                    <li>• FastAPI backend with real-time inference</li>
                    <li>• React frontend with interactive visualizations</li>
                  </ul>
                  
                  <h3 className="text-xl font-semibold mb-4 mt-6 text-orange-600">📊 Customer Segments</h3>
                  <ul className="text-black space-y-1 text-sm">
                    <li>• Urban Youth • Premium Urban • Budget Conscious</li>
                    <li>• Social Millennials • Corporate Professional</li>
                    <li>• Loyal Veterans • Weekend Warriors</li>
                    <li>• Price Seekers • Premium Explorers</li>
                    <li>• Rural Traditional</li>
                  </ul>
                </div>
              </div>
              
              <div className="mt-8 p-6 bg-gradient-to-r from-blue-50 to-green-50 rounded-lg force-black">
                <h3 className="text-lg font-semibold mb-2">🏆 Success Criteria</h3>
                <div className="grid md:grid-cols-3 gap-4 text-sm">
                  <div>
                    <strong>Model Performance:</strong><br />
                    Silhouette score &gt; 0.7
                  </div>
                  <div>
                    <strong>App Usability:</strong><br />
                    95% uptime, intuitive UI
                  </div>
                  <div>
                    <strong>Portfolio Impact:</strong><br />
                    GitHub stars, demo satisfaction
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
