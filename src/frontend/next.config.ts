import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  reactStrictMode: false,
  images: {
    domains: ["developers.google.com"],
    remotePatterns: [
      {
        protocol: "https",
        hostname: "cdn.prod.website-files.com",
        pathname: "/**",
      },
    ],
  },
};

export default nextConfig;
