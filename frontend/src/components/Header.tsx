"use client";
import { useEffect, useState } from 'react';
import { Github, ExternalLink, Sun, Moon } from 'lucide-react';

export default function Header() {
  const [theme, setTheme] = useState<'light'|'dark'>(() => {
    if (typeof window === 'undefined') return 'light';
    return (localStorage.getItem('theme') as 'light'|'dark') || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  });

  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'dark') root.classList.add('dark');
    else root.classList.remove('dark');
    localStorage.setItem('theme', theme);
  }, [theme]);

  return (
    <header className="sticky top-0 z-40 backdrop-blur bg-white/80 border-b">
      <div className="container mx-auto px-4 py-3">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="text-2xl font-bold text-slate-800 tracking-tight">🍺 EABL Insights</div>
            <div className="hidden md:block text-sm text-slate-500">Customer Segmentation Challenge</div>
          </div>
          {/* Right side intentionally left clean per request */}
          <div />
        </div>
      </div>
    </header>
  );
}
