import React from 'react';

type ClusterSummary = {
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

interface CountyTableProps {
  clusterSummary: ClusterSummary[];
}

// Note: This aggregates counts from each segment's top_counties only.
// It is an estimate until we compute exact county totals during inference.
export default function CountyTable({ clusterSummary }: CountyTableProps) {
  const countyTotals: Record<string, { total: number; topSegment: string; topCount: number }> = {};

  for (const seg of clusterSummary) {
    Object.entries(seg.top_counties || {}).forEach(([county, count]) => {
      if (!countyTotals[county]) {
        countyTotals[county] = { total: 0, topSegment: seg.cluster_name, topCount: count };
      }
      countyTotals[county].total += count;
      if (count > countyTotals[county].topCount) {
        countyTotals[county].topSegment = seg.cluster_name;
        countyTotals[county].topCount = count;
      }
    });
  }

  const rows = Object.entries(countyTotals)
    .map(([county, info]) => ({ county, total: info.total, topSegment: info.topSegment }))
    .sort((a, b) => b.total - a.total)
    .slice(0, 20); // show top 20

  if (rows.length === 0) {
    return <p className="text-sm text-slate-600">No county data available in the current results.</p>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="text-left text-slate-700 dark:text-slate-200">
            <th className="py-2 pr-4">County</th>
            <th className="py-2 pr-4">Estimated Customers</th>
            <th className="py-2 pr-4">Top Segment</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.county} className="border-t border-[var(--border)]">
              <td className="py-2 pr-4 text-slate-800 dark:text-slate-100">{r.county}</td>
              <td className="py-2 pr-4 text-slate-800 dark:text-slate-100">{r.total.toLocaleString()}</td>
              <td className="py-2 pr-4 text-slate-600 dark:text-slate-300">{r.topSegment}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-xs text-slate-500 mt-2">Note: County totals are aggregated from each segment's Top Counties list and approximate the true distribution.</p>
    </div>
  );
}
