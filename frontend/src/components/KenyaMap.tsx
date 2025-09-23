"use client";

import React, { useEffect, useMemo, useState } from 'react';
import dynamic from 'next/dynamic';

// Use Plotly dynamically to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

export type ClusterSummary = {
  cluster_id: number;
  cluster_name: string;
  size: number;
  percentage: number;
  avg_age: number;
  avg_engagement: number;
  avg_purchases: number;
  top_counties: Record<string, number>;
  top_behaviors: Record<string, number>;
};

interface KenyaMapProps {
  clusterSummary: ClusterSummary[];
  metric: 'customers' | 'share' | 'engagement';
  segmentName?: string; // if provided, filter to this segment
}

// Attempt to fetch a local/remote Kenya counties GeoJSON. If unavailable, show a helpful message.
const KENYA_GEOJSON_URLS = [
  // Local first (place file at: public/data/kenya-counties.geojson)
  '/data/kenya-counties.geojson',
  // Public mirrors
  // geoBoundaries (primary official repo)
  'https://raw.githubusercontent.com/wmgeolab/geoBoundaries/main/releaseData/gbOpen/KEN/ADM1/geoBoundaries-KEN-ADM1.geojson',
  // jsDelivr mirror of the same
  'https://cdn.jsdelivr.net/gh/wmgeolab/geoBoundaries@main/releaseData/gbOpen/KEN/ADM1/geoBoundaries-KEN-ADM1.geojson',
  // Alternate mirror
  'https://raw.githubusercontent.com/statoood/kenya-geojson/master/kenya-counties.geojson'
];

export default function KenyaMap({ clusterSummary, metric, segmentName }: KenyaMapProps) {
  const [geojson, setGeojson] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    (async () => {
      for (const url of KENYA_GEOJSON_URLS) {
        try {
          const res = await fetch(url);
          if (!res.ok) continue;
          const gj = await res.json();
          if (isMounted) {
            setGeojson(gj);
            setError(null);
            return;
          }
        } catch (e) {
          // try next
        }
      }
      if (isMounted) setError('Unable to load Kenya GeoJSON. Please check your network or add a local file at /public/data/kenya-counties.geojson.');
    })();
    return () => { isMounted = false; };
  }, []);

  // Build a county -> value map from clusterSummary
  const { countyNames, countyValues } = useMemo(() => {
    // Aggregate: customers per county using top_counties (approximation)
    const totals: Record<string, number> = {};

    const segments = segmentName
      ? clusterSummary.filter(s => s.cluster_name === segmentName)
      : clusterSummary;

    for (const seg of segments) {
      Object.entries(seg.top_counties || {}).forEach(([county, count]) => {
        totals[county] = (totals[county] || 0) + (count || 0);
      });
    }

    const names = Object.keys(totals);
    const values = names.map(n => totals[n]);
    return { countyNames: names, countyValues: values };
  }, [clusterSummary, segmentName, metric]);

  // Fallback scattergeo bubble map if geojson is unavailable
  if (error || !geojson) {
    const centroids: Record<string, { lon: number; lat: number }> = {
      'Nairobi': { lon: 36.8219, lat: -1.2921 },
      'Mombasa': { lon: 39.6682, lat: -4.0435 },
      'Kisumu': { lon: 34.7617, lat: -0.0917 },
      'Nakuru': { lon: 36.0667, lat: -0.2833 },
      'Eldoret': { lon: 35.2698, lat: 0.5143 },
      'Thika': { lon: 37.0707, lat: -1.0333 },
      'Malindi': { lon: 40.1191, lat: -3.2192 },
      'Kitale': { lon: 34.9930, lat: 1.0157 },
      'Garissa': { lon: 39.6583, lat: -0.4531 },
      'Kakamega': { lon: 34.7519, lat: 0.2827 },
      'Bungoma': { lon: 34.5645, lat: 0.5695 },
      'Kericho': { lon: 35.2886, lat: -0.3689 },
      'Kiambu': { lon: 36.8336, lat: -1.1714 },
      'Machakos': { lon: 37.2634, lat: -1.5167 },
      'Kajiado': { lon: 36.7785, lat: -1.8521 },
      'Nyeri': { lon: 36.9510, lat: -0.4201 },
      'Meru': { lon: 37.6525, lat: 0.0463 },
      'Embu': { lon: 37.4500, lat: -0.5333 },
      'Narok': { lon: 35.8711, lat: -1.0856 },
      'Uasin Gishu': { lon: 35.3426, lat: 0.5540 },
      'Turkana': { lon: 35.4786, lat: 3.1219 },
      'Kilifi': { lon: 39.8582, lat: -3.6333 }
    };

    const segments = segmentName
      ? clusterSummary.filter(s => s.cluster_name === segmentName)
      : clusterSummary;

    const totals: Record<string, number> = {};
    for (const seg of segments) {
      Object.entries(seg.top_counties || {}).forEach(([county, count]) => {
        totals[county] = (totals[county] || 0) + (count || 0);
      });
    }

    const names: string[] = [];
    const lons: number[] = [];
    const lats: number[] = [];
    const sizes: number[] = [];

    Object.entries(totals).forEach(([county, value]) => {
      const c = centroids[county];
      if (c && value > 0) {
        names.push(county);
        lons.push(c.lon);
        lats.push(c.lat);
        sizes.push(Math.max(6, Math.sqrt(value)));
      }
    });

    if (names.length === 0) {
      return (
        <div className="p-4 bg-amber-50 border border-amber-200 rounded text-sm text-amber-800">
          <strong>Map data note:</strong> No matching counties found for a quick bubble map. Please add the local GeoJSON at
          <code> /public/data/kenya-counties.geojson</code> for full county polygons.
        </div>
      );
    }

    return (
      <div className="w-full h-[520px]">
        <Plot
          data={[{
            type: 'scattergeo',
            lon: lons,
            lat: lats,
            text: names.map((n, i) => `${n}: ${totals[n].toLocaleString()} customers`),
            marker: {
              size: sizes,
              color: sizes,
              colorscale: [[0, '#C7D2FE'], [1, '#4F46E5']],
              line: { width: 0 }
            },
            hovertemplate: '%{text}<extra></extra>'
          }]}
          layout={{
            geo: {
              scope: 'africa',
              showframe: false,
              showcoastlines: false,
              projection: { type: 'mercator' },
              center: { lon: 37.9, lat: 0.0 },
              lonaxis: { range: [33, 42] },
              lataxis: { range: [-5, 5] },
              bgcolor: 'rgba(0,0,0,0)'
            },
            margin: { t: 20, b: 0, l: 0, r: 0 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
          }}
          config={{ displayModeBar: false }}
          style={{ width: '100%', height: '100%' }}
        />
      </div>
    );
  }

  // Prepare Plotly choropleth
  // We'll map county names from our data to geojson feature properties. Many datasets use NAME_1 or COUNTY_NAM.
  // Attempt to detect a name property.
  const featureNameProp = detectCountyNameProp(geojson);

  const featureIds: string[] = (geojson.features || []).map((f: any) => sanitize((f.properties?.[featureNameProp] || '') as string));
  const zValues: number[] = featureIds.map(id => {
    const idx = countyNames.map(sanitize).indexOf(id);
    return idx >= 0 ? countyValues[idx] : 0;
  });

  // Indigo scale
  const colorscale = [
    [0, '#EEF2FF'],
    [0.2, '#C7D2FE'],
    [0.4, '#A5B4FC'],
    [0.6, '#818CF8'],
    [0.8, '#6366F1'],
    [1, '#4F46E5']
  ];

  const title = metric === 'customers'
    ? 'Customers per County'
    : metric === 'share'
    ? 'Segment Share (%) per County'
    : 'Average Engagement per County';

  const hover = metric === 'customers'
    ? '%{location}<br>%{z:,} customers<extra></extra>'
    : metric === 'share'
    ? '%{location}<br>%{z:.1f}% share<extra></extra>'
    : '%{location}<br>%{z:.2f} engagement<extra></extra>';

  return (
    <div className="w-full h-[520px]">
      <Plot
        data={[{
          type: 'choropleth',
          geojson,
          locations: featureIds,
          z: zValues,
          featureidkey: `properties.${featureNameProp}`,
          colorscale,
          marker: { line: { width: 0 } },
          hovertemplate: hover,
          colorbar: { title: metric === 'customers' ? 'Customers' : metric === 'share' ? 'Share %' : 'Engagement' }
        }]}
        layout={{
          geo: {
            scope: 'world',
            showframe: false,
            showcoastlines: false,
            projection: { type: 'mercator' },
            lonaxis: { range: [33, 42.5] },
            lataxis: { range: [-5, 5.5] },
            bgcolor: 'rgba(0,0,0,0)'
          },
          margin: { t: 40, b: 0, l: 0, r: 0 },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          title: { text: title, x: 0, font: { size: 16 } },
          showlegend: false
        }}
        config={{ displayModeBar: false }}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
}

function sanitize(name: string) {
  return name.trim().toLowerCase().replace(/[^a-z0-9]+/g, '');
}

function detectCountyNameProp(gj: any): string {
  const props = gj?.features?.[0]?.properties || {};
  const candidateKeys = ['COUNTY_NAM', 'COUNTY', 'NAME_1', 'name', 'Name', 'ADM1_EN'];
  for (const k of candidateKeys) {
    if (k in props) return k;
  }
  // fallback to first key
  return Object.keys(props)[0] || 'name';
}
