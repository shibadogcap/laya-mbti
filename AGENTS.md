# AGENTS.md — laya-mbti

Guidance for AI coding agents working in this repository. Kiro also reads
`.kiro/steering/` for the same conventions.

## Project

Static, client-only web app that estimates MBTI tendencies from an X (Twitter)
archive using the Laya typed-decision model in the browser.

## Commands

- `pnpm install` — install (pnpm only; never npm/yarn)
- `pnpm dev` — dev server at `127.0.0.1:5173`
- `pnpm dev:https` — self-signed HTTPS dev server for LAN (WebGPU needs a secure context)
- `pnpm typecheck` — TypeScript check
- `pnpm test` — Vitest unit tests for `src/lib`
- `pnpm build` — production build
- `pnpm build:pages` — build + model checksum / 1 GB budget gate
- `pnpm model:export` — export the pinned Laya checkpoint (needs `uv`)

## Rules

1. **pnpm only.** Never introduce `npm`, `yarn`, or `package-lock.json`.
2. **Client-only.** Never send user content off-origin. Allowed network fetches
   are model assets and ONNX Runtime Wasm from our own origin.
3. **Workers for heavy work.** Archive parsing and inference run in
   `src/workers/`. Keep the main thread free.
4. **No large files in git.** `public/models/**` is gitignored and generated in
   CI. Never stage model binaries.
5. **Keep the deploy under 1 GB.** Do not weaken `scripts/validate-pages.mjs`.
6. **Preserve attribution.** Vendored `r4ai/laya-web` files stay Apache-2.0 and
   are recorded in `NOTICE`.
7. **Pure logic stays testable.** `src/lib/parse.ts`, `filter.ts`, and
   `aggregate.ts` must not import browser-only APIs.

## Layout

See `.kiro/steering/structure.md`. Feature intent and math are in
`.kiro/specs/mbti-analysis/`.
