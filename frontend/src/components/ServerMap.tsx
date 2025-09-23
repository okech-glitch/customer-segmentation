"use client";

import React, { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import { api } from '@/lib/api';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface ServerMapProps {
  metric: 'customers' | 'share' | 'engagement';
  segment: string; // 'All' or cluster name
}

export default function ServerMap({ metric, segment }: ServerMapProps) {
  const [fig, setFig] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setLoading(true);
        setError(null);
        setFig(null);
        const res = await api.get('/map/figure', { params: { metric, segment } });
        if (!alive) return;
        setFig(res.data);
      } catch (e: any) {
        if (!alive) return;
        setError(e?.response?.data?.detail || 'Failed to load map figure');
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [metric, segment]);

  if (loading) return <div className="text-sm text-slate-600">Loading map…</div>;
  if (error) return <div className="p-3 bg-amber-50 border border-amber-200 rounded text-amber-800 text-sm">{error}</div>;
  if (!fig) return null;

  return (
    <div className="w-full h-[520px]">
      <Plot
        data={fig.data || []}
        layout={{ ...fig.layout, autosize: true }}
        config={{ displayModeBar: false }}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
}
