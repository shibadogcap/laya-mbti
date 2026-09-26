import { createReadStream } from "node:fs";
import { readFile } from "node:fs/promises";
import basicSsl from "@vitejs/plugin-basic-ssl";
import { defineConfig, type Plugin } from "vite";
import solid from "vite-plugin-solid";

const repoRoot = new URL("./", import.meta.url);
const ortDist = (name: string) =>
  new URL(`node_modules/onnxruntime-web/dist/${name}`, repoRoot);

// ONNX Runtime Web resolves these by name under wasmPaths, and its JS and Wasm
// must come from the same version. Ship both the WebGPU (jsep) and the
// Wasm-only (asyncify) pairs so `backend: "auto"` can pick either.
const ortFiles = [
  "ort-wasm-simd-threaded.asyncify.mjs",
  "ort-wasm-simd-threaded.asyncify.wasm",
  "ort-wasm-simd-threaded.jsep.mjs",
  "ort-wasm-simd-threaded.jsep.wasm",
];

/** Serves the ONNX Runtime Web binaries from `/ort/` in dev and emits them on build. */
const selfHostOrt = (): Plugin => ({
  name: "self-host-ort",
  configureServer(server) {
    server.middlewares.use("/ort", (req, res, next) => {
      const name = req.url?.slice(1).split("?")[0] ?? "";
      if (!ortFiles.includes(name)) return next();

      res.setHeader(
        "Content-Type",
        name.endsWith(".wasm") ? "application/wasm" : "text/javascript",
      );
      createReadStream(ortDist(name)).on("error", next).pipe(res);
    });
  },
  async generateBundle(_options, bundle) {
    // ORT's ESM imports reference the Wasm binary, so Vite emits a second copy
    // under assets/. We serve the runtime from /ort/ via wasmPaths, so drop the
    // duplicate to stay inside the GitHub Pages 1 GB size budget.
    for (const key of Object.keys(bundle)) {
      if (/^assets\/ort-wasm-.*\.wasm$/.test(key)) delete bundle[key];
    }
    for (const name of ortFiles)
      this.emitFile({
        type: "asset",
        fileName: `ort/${name}`,
        source: await readFile(ortDist(name)),
      });
  },
});

const cacheStaticAssets = (): Plugin => ({
  name: "cache-static-assets",
  configurePreviewServer(server) {
    server.middlewares.use((req, res, next) => {
      const path = req.url?.split("?")[0] ?? "";
      if (path.startsWith("/models/") || path.startsWith("/ort/")) {
        res.setHeader("Cache-Control", "public, max-age=31536000, immutable");
      }
      next();
    });
  },
});

export default defineConfig({
  base: "./",
  worker: { format: "es" },
  // The review tunnel reaches the dev server through a `*.trycloudflare.com`
  // hostname, which changes every time it is recreated. Vite blocks unknown
  // hosts by default, so allow the tunnel host and its subdomains.
  server: { allowedHosts: [".trycloudflare.com"] },
  preview: { allowedHosts: true },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  // WebGPU needs a secure context. `pnpm dev` stays on http://localhost; `pnpm
  // dev:https` adds a self-signed certificate so LAN IPs can use WebGPU too.
  plugins: [
    solid(),
    selfHostOrt(),
    cacheStaticAssets(),
    ...(process.env.VITE_HTTPS ? [basicSsl()] : []),
  ],
});
