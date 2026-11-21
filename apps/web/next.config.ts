import type { NextConfig } from "next";

function apiProxyTarget() {
  const configured =
    process.env.NEXT_SERVER_API_BASE_URL ||
    process.env.NEXT_PUBLIC_API_BASE_URL ||
    "http://127.0.0.1:8000";
  return configured.replace(/\/$/, "");
}

const nextConfig: NextConfig = {
  // 允许本机 127.0.0.1 / localhost 及局域网地址在开发模式下加载 /_next 资源
  allowedDevOrigins: ["127.0.0.1", "localhost", "172.27.2.90", "100.94.222.54"],
  experimental: {
    // 允许上传大视频文件（后端限制 4GB，这里设 500MB 给代理留余量）
    proxyClientMaxBodySize: "2000mb",
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${apiProxyTarget()}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
