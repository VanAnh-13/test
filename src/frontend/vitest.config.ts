import { createRequire } from "node:module";
import { fileURLToPath, URL } from "node:url";

import { defineConfig } from "vitest/config";

const loadModule = createRequire(import.meta.url);
const react = loadModule("@vitejs/plugin-react").default;

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.tsx", "src/**/*.vitest.ts"],
    globals: false,
    clearMocks: true,
    restoreMocks: true,
  },
});
