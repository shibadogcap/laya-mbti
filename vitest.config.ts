import { defineConfig } from "vitest/config";

// Tests cover the pure logic in src/lib and must not pull in the browser-only
// Vite plugins (solid, ORT self-hosting), so this config intentionally stands
// alone from vite.config.ts.
export default defineConfig({
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
  },
});
