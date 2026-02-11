/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static export for serverless deployment (GitHub Pages, etc.)
  output: 'export',

  // Disable image optimization for static export
  images: {
    unoptimized: true,
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**.github.io',
      },
      {
        protocol: 'https',
        hostname: 'openi.nlm.nih.gov',
      },
    ],
  },

  // Trailing slash for static hosting compatibility
  trailingSlash: true,
};

module.exports = nextConfig;
