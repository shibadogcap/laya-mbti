---
name: verify-model
description: Verify the exported Laya model assets and the Pages size budget before deploying. Use when model assets, the export pipeline, or deployment config change.
---

# Verify model and Pages budget

## When to use

- After changing `scripts/export_laya_ts.py`, `pyproject.toml`, or `uv.lock`.
- Before a deploy, or when the model cache key in
  `.github/workflows/pages.yml` changes.
- When the app reports a model checksum or format error at load time.

## Steps

1. Ensure the model exists locally:

   ```sh
   ls -la public/models/laya
   ```

   If missing, export it (requires `uv`, ~644 MB checkpoint download):

   ```sh
   pnpm model:export
   ```

2. Run the deploy gate, which recomputes SHA-256 for every file listed in
   `config.json` and enforces the 1 GB budget:

   ```sh
   pnpm build:pages
   ```

3. Confirm the expected sizes are present:

   | File | Approx size |
   | --- | --- |
   | `model.onnx.data` | 501 MB |
   | `embeddings.f16.bin` | 393 MB |
   | `tokenizer/tokenizer.json` | 34 MB |
   | `ort/*.wasm` (two files) | 55 MB |

4. If a checksum fails, delete `public/models/laya` and re-export from the
   pinned revision. Never edit `config.json` by hand.

## Failure modes

- **`Unsupported model format`** — the export was produced by a different
  exporter version; re-export.
- **Size over budget** — a duplicate Wasm asset or source map crept in; check
  the `self-host-ort` plugin in `vite.config.ts`.
- **Missing tokenizer files** — the Hugging Face snapshot was incomplete;
  re-export with network access.
