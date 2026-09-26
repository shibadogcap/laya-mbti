# Tech Steering — laya-mbti

## Stack

- **Language:** TypeScript, strict mode (`noUnusedLocals`, `verbatimModuleSyntax`).
- **UI:** SolidJS + Vite. `vite-plugin-solid` handles JSX.
- **Package manager:** **pnpm only.** Never run `npm` or `yarn`; never commit a
  `package-lock.json`. The lockfile is `pnpm-lock.yaml`.
- **Inference:** official `laya-ts` (vendored under `src/vendor/laya-ts/`) with
  `onnxruntime-web` (pinned `1.30.0`). The encoder runs on the WebGPU execution
  provider, with WASM as the fallback provider; WASM also backs the small head
  session on the WebGPU path.
- **ZIP reading:** `@zip.js/zip.js`, streamed inside a dedicated Web Worker.
- **Model export:** Python via `uv` (`scripts/export_laya_ts.py`, the `laya`
  package), CPU-only torch. The encoder is exported in fp16 with an fp32 output;
  the head stays fp32.

## Commands

| Command | Purpose |
| --- | --- |
| `pnpm dev` | Vite dev server on `127.0.0.1:5173` (http) |
| `pnpm dev:https` | Self-signed HTTPS dev server for LAN (WebGPU needs a secure context) |
| `pnpm typecheck` | `tsc --noEmit` |
| `pnpm build` | Production build into `dist/` |
| `pnpm build:pages` | Build + `scripts/validate-pages.mjs` size/integrity gate |
| `pnpm model:export` | Export the pinned checkpoint into `public/models/laya` |

## Constraints

- **Everything runs in the browser.** Do not add a server, API routes, or
  network calls that transmit user content. The only network fetches allowed are
  model assets and ONNX Runtime Wasm files from our own origin.
- **No large files in git.** `public/models/**` is gitignored and generated in
  CI. Never `git add` model binaries.
- **Keep the UI thread free.** Heavy work (unzip, inference) belongs in Web
  Workers. Do not run model inference or JSON parsing of archives on the main
  thread.
- **WebGPU first, WASM as the fallback.** The encoder prefers the WebGPU
  execution provider. When the fp16 preflight refuses the GPU, when a WebGPU
  session cannot be created, or when a WebGPU batch dies at runtime, the worker
  rebuilds the provider on WASM and restarts the analysis from the first tweet
  (`decideWasmFallback` / `nextBackend` in `src/vendor/laya-ts/providers.ts`).
  A partial WebGPU result is never presented: the `restart` message drops the
  items already sent. Say which backend is running in the status line — the CPU
  path is far slower, so the user has to be able to lower the sample size.
  WebGPU only exists in a secure context (https or localhost), so LAN IP over
  plain http has no `navigator.gpu` and runs on the CPU; use `pnpm dev:https`
  there for the fast path.
- **Respect the GitHub Pages budget.** The published site must stay under 1 GB.
  `pnpm build:pages` enforces this; do not remove the check.
- **Attribute upstream.** Vendored `laya-ts` files stay Apache-2.0, keep their
  modification notes, and are recorded in `NOTICE`.

## Style

- Prefer small, pure functions in `src/lib/` and keep components presentational.
- No comments unless they explain non-obvious intent.
- Type-only imports use `import type`.
