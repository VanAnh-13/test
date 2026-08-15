import { randomBytes } from "node:crypto";

import { defineConfig, devices } from "@playwright/test";

const port = Number.parseInt(process.env.PLAYWRIGHT_PORT ?? "3000", 10);

if (!Number.isInteger(port) || port < 1 || port > 65_535) {
  throw new Error("PLAYWRIGHT_PORT phải là cổng TCP hợp lệ");
}

const localBaseUrl = `http://127.0.0.1:${port}`;
const sessionSecret = randomBytes(32).toString("hex");
const standaloneCommand = [
  "node -e \"const fs=require('node:fs');",
  "fs.cpSync('public','.next/standalone/public',{recursive:true});",
  "fs.cpSync('.next/static','.next/standalone/.next/static',{recursive:true})\"",
  "&& node .next/standalone/server.js",
].join(" ");

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: "list",
  use: {
    baseURL: localBaseUrl,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: {
    command: standaloneCommand,
    url: localBaseUrl,
    env: {
      HOSTNAME: "127.0.0.1",
      NEXTAUTH_SECRET: sessionSecret,
      NEXTAUTH_URL: localBaseUrl,
      PORT: String(port),
    },
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
