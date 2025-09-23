import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    unoptimized: true,
  },
  // Fix for multiple lockfiles warning
  experimental: {
    outputFileTracingRoot: process.cwd(),
  } as any,
};

export default nextConfig;
