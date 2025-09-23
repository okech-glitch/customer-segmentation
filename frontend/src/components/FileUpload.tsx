'use client';

import { useState, useCallback } from 'react';
import { Upload, Loader2, CheckCircle, AlertCircle, Database } from 'lucide-react';
import { api, getApiBaseUrl, normalizeAxiosError } from '@/lib/api';

interface FileUploadProps {
  onSegmentationComplete: (results: any) => void;
}

export default function FileUpload({ onSegmentationComplete }: FileUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sampleData, setSampleData] = useState<any>(null);

  // Dynamic API base for localhost or LAN IP access
  const API_BASE_URL = getApiBaseUrl();

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type === 'text/csv' || droppedFile.name.endsWith('.csv')) {
        setFile(droppedFile);
        setError(null);
      } else {
        setError('Please upload a CSV file');
      }
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      if (selectedFile.type === 'text/csv' || selectedFile.name.endsWith('.csv')) {
        setFile(selectedFile);
        setError(null);
      } else {
        setError('Please upload a CSV file');
      }
    }
  };

  const generateSampleData = async () => {
    try {
      setUploading(true);
      setError(null);
      const response = await api.post(`/generate-data`, null, {
        params: { n_samples: 1000 }
      });
      setSampleData(response.data);
    } catch (err: any) {
      console.error('Generate data error:', err);
      setError(normalizeAxiosError(err));
    } finally {
      setUploading(false);
    }
  };

  const uploadFile = async () => {
    if (!file) return;

    try {
      setUploading(true);
      setError(null);

      const formData = new FormData();
      formData.append('file', file);

      const response = await api.post(`/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      onSegmentationComplete(response.data);
    } catch (err: any) {
      setError(normalizeAxiosError(err));
    } finally {
      setUploading(false);
    }
  };

  const downloadSample = async () => {
    try {
      const response = await api.get(`/download-sample`, {
        responseType: 'blob',
      });
      
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'sample_data.csv');
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      setError('Failed to download sample data');
    }
  };

  const downloadTest = async () => {
    try {
      const response = await api.get(`/download-test`, { responseType: 'blob' });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'test_data.csv');
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      setError('Failed to download test data. Try generating data first.');
    }
  };

  const downloadTrain = async () => {
    try {
      const response = await api.get(`/download-train`, { responseType: 'blob' });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'train_data.csv');
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      setError('Failed to download train data. Try generating data first.');
    }
  };

  return (
    <div className="space-y-8">
      {/* Upload Section */}
      <div className="card p-8">
        <h2 className="text-2xl font-bold mb-6 text-gray-800">Upload Customer Data</h2>
        
        <div
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
            dragActive
              ? 'border-blue-400 bg-blue-50'
              : file
              ? 'border-green-400 bg-green-50'
              : 'border-gray-300 hover:border-gray-400'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          {file ? (
            <div className="space-y-4">
              <CheckCircle className="w-12 h-12 text-green-500 mx-auto" />
              <div>
                <p className="text-lg font-medium text-gray-800">{file.name}</p>
                <p className="text-sm text-gray-500">
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
              <button
                onClick={uploadFile}
                disabled={uploading}
                className="btn btn-secondary disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2 mx-auto"
              >
                {uploading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>Processing...</span>
                  </>
                ) : (
                  <>
                    <Upload className="w-5 h-5" />
                    <span>Analyze Customer Segments</span>
                  </>
                )}
              </button>
            </div>
          ) : (
            <div className="space-y-4">
              <Upload className="w-12 h-12 text-gray-400 mx-auto" />
              <div>
                <p className="text-lg font-medium text-gray-600">
                  Drop your CSV file here, or{' '}
                  <label className="text-blue-600 hover:text-blue-700 cursor-pointer underline">
                    browse
                    <input
                      type="file"
                      accept=".csv"
                      onChange={handleFileChange}
                      className="hidden"
                    />
                  </label>
                </p>
                <p className="text-sm text-gray-500 mt-2">
                  Supports CSV files up to 50MB
                </p>
              </div>
            </div>
          )}
        </div>

        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-start space-x-2">
              <AlertCircle className="w-5 h-5 text-red-500 mt-0.5" />
              <div className="text-red-700 text-sm">
                <div className="font-medium">{error}</div>
                {error.includes('Missing required columns') && (
                  <div className="mt-2">
                    <div className="text-xs text-red-600">Use the full test data template which includes all required columns.</div>
                    <button onClick={downloadTest} className="btn btn-primary mt-2 flex items-center space-x-2">
                      <Database className="w-4 h-4" />
                      <span>Download Full Test CSV</span>
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Sample Data Section */}
      <div className="card p-8">
        <h2 className="text-2xl font-bold mb-6 text-gray-800">Get Test Data</h2>
        <p className="text-gray-600 mb-4">Download the full test CSV that includes all required columns. Use this to try the segmentation quickly.</p>
        <button onClick={downloadTest} className="btn btn-primary flex items-center space-x-2">
          <Database className="w-5 h-5" />
          <span>Download Full Test CSV</span>
        </button>

        {sampleData && (
          <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
            <div className="flex items-center space-x-2 mb-2">
              <CheckCircle className="w-5 h-5 text-green-500" />
              <span className="font-medium text-green-800">Sample Data Generated!</span>
            </div>
            <p className="text-green-700 text-sm">
              Generated {sampleData.total_records?.toLocaleString()} customer records with {sampleData.columns?.length} features.
            </p>
            <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
              {sampleData.columns?.slice(0, 8).map((col: string) => (
                <div key={col} className="bg-white px-2 py-1 rounded text-gray-700">
                  • {col}
                </div>
              ))}
            </div>
            <p className="text-green-600 text-xs mt-2">
              Data is ready to test the segmentation model!
            </p>
          </div>
        )}
      </div>

      {/* Data Requirements */}
      <div className="card p-8">
        <h2 className="text-2xl font-bold mb-6 text-gray-800">Data Requirements</h2>
        
        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <h3 className="text-lg font-semibold mb-4 text-blue-600">Required Columns</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center p-2 chip-light force-black">
                <span className="chip-left">customer_id</span>
                <span className="chip-right">Unique identifier</span>
              </div>
              <div className="flex justify-between items-center p-2 chip-light force-black">
                <span className="chip-left">age</span>
                <span className="chip-right">Customer age (18-70)</span>
              </div>
              <div className="flex justify-between items-center p-2 chip-light force-black">
                <span className="chip-left">income_kes</span>
                <span className="chip-right">Income bracket</span>
              </div>
              <div className="flex justify-between items-center p-2 chip-light force-black">
                <span className="chip-left">purchase_history</span>
                <span className="chip-right">Transaction count</span>
              </div>
              <div className="flex justify-between items-center p-2 chip-light force-black">
                <span className="chip-left">behavior</span>
                <span className="chip-right">Product preferences</span>
              </div>
            </div>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-4 text-green-600">Additional Fields</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center p-2 chip-light force-black">
                <span className="chip-left">engagement_rate</span>
                <span className="chip-right">Marketing (0-1)</span>
              </div>
              <div className="flex justify-between items-center p-2 chip-light force-black">
                <span className="chip-left">county</span>
                <span className="chip-right">Kenyan county</span>
              </div>
              <div className="flex justify-between items-center p-2 chip-light force-black">
                <span className="chip-left">profit_segment</span>
                <span className="chip-right">EABL category</span>
              </div>
              <div className="flex justify-between items-center p-2 chip-light force-black">
                <span className="chip-left">upgrade_engagement</span>
                <span className="chip-right">M-Pesa boost</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
