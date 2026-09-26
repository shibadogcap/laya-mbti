# Tasks — MBTI estimation from an X archive

- [x] 1. Scaffold Vite + SolidJS + TypeScript with pnpm.
- [x] 2. Add `onnxruntime-web` and `@zip.js/zip.js` at pinned versions; vendor the
  official `laya-ts` runtime.
- [x] 3. Implement pure archive parsers in `src/lib/parse.ts` (new `tweets.js`,
  legacy `tweets.csv`, `account.js`).
- [x] 4. Implement `archive.worker.ts` with streaming zip.js reads and
  cross-archive dedupe.
- [x] 5. Implement `inference.worker.ts` (Laya load + per-tweet predict loop,
  progress, cancel).
- [x] 6. Define axes and the question batch in `src/lib/mbti.ts`.
- [x] 7. Implement pre-filtering (`filter.ts`) and confidence-weighted
  aggregation (`aggregate.ts`).
- [x] 8. Build the UI: drop zone, period picker, settings, progress, result card.
- [x] 9. Self-host ONNX Runtime Wasm in `vite.config.ts` and drop the duplicate
  Wasm asset.
- [x] 10. Add `scripts/validate-pages.mjs` (checksums + 1 GB budget).
- [x] 11. Add the GitHub Pages workflow with model export caching.
- [x] 12. Verify inference and ZIP parsing end-to-end (Node harness; browser with a real archive still pending).
- [x] 13. Add automated unit tests for `parse.ts` and `aggregate.ts`.
- [x] 14. Add a Service Worker cache for model reuse and parallel workers.
- [ ] 15. Record the demo video and publish the challenge submission.
