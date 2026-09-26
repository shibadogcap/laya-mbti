# Design — MBTI estimation from an X archive

## Data flow

```
File[] ──▶ archive.worker
             │  ZipReader(BlobReader, { useWebWorkers: false })
             │  └─ only data/tweets*.{js,csv}, data/account*.js
             │     parse → Map<id, Tweet{id, createdAt, text}>
             ▼
         ArchiveSummary ──▶ App state
             │  filterByPeriod → preFilter (pure, src/lib/filter.ts)
             ▼
         { id, text }[] ──▶ inference.worker
             │  Laya.load(modelUrl, wasmPaths, "auto")
             │  for each tweet: agent.predict(text, buildQuestions())
             ▼
         { id, answers }[] ──▶ aggregate (pure, src/lib/aggregate.ts)
             ▼
         MbtiResult ──▶ ResultCard
```

## Key decisions

### Why zip.js with `useWebWorkers: false`

The archive worker is already off the main thread, so we disable zip.js's own
worker pool. This keeps bundling simple and avoids nested-worker issues while
still not blocking the UI. zip.js supports Zip64, which matters because large
archives with many media entries exceed the classic ZIP limits.

### Why one worker per concern

`archive.worker` and `inference.worker` are separate so the model session stays
resident while the user loads more archives, and so a long inference run can be
cancelled without tearing down the model.

### Question batch

Per tweet, four Laya questions run against the same state — one two-way
`choice` per axis:

| id | type | options |
| --- | --- | --- |
| `EI` | `choice` | E / I |
| `SN` | `choice` | S / N |
| `TF` | `choice` | T / F |
| `JP` | `choice` | J / P |

Axis definitions live in `src/lib/mbti.ts`.

**Why four questions.** Calibration runs against the pinned checkpoint showed
that:

- a three-way `choice` (positive / negative / neutral) dilutes the probability
  and collapses the normalized-entropy confidence;
- two independent `noul` questions per axis were more directional but doubled
  the passes (8 per tweet);
- a **two-way `choice`** keeps the margin between the poles and is just as
  stable, at one pass per axis (4 per tweet).

The official `laya-ts` runtime collates all four questions as rows in a single
batch, so they cost one encoder forward pass, not four.

### Aggregation math

For each axis and tweet `t`, with two-way probabilities `p⁺` and `p⁻`:

```
margin = p⁺ - p⁻
if margin > 0: positive += margin   else: negative += -margin
```

Tweets with `|margin| < marginThreshold` are counted as retained but do not move
the axis. The winning pole is the one with the larger accumulated magnitude;
ties resolve to the positive pole. Reported probability is
`max(pos, neg) / (pos + neg)`.

### Live progress

The worker posts each analyzed tweet as an `item` message as soon as it is
ready. The main thread appends it and recomputes the aggregate on a 250 ms
throttle, so a provisional four-letter type is visible during a long run. The
final aggregate runs once when the worker posts `done`.

### Sampling

Inference still costs roughly a second per batched pass, so the UI caps the
analysis at `maxTweets` (default 500) and samples evenly across the selected
period rather than taking only the newest tweets. Setting it to 0 analyzes
everything.

### Batching

The official `laya-ts` runtime batches questions as **rows in the batch
dimension** (`collateItems`), with `attention_mask` handling padding. We use its
`predictBatch` to pack up to 64 rows (16 tweets × 4 questions) into one encoder
forward pass, so tweets are judged in parallel on the GPU. The exported graph's
dynamic batch dimension is capped at 64; the worker starts at 16 tweets per
batch and halves on a memory error.

An earlier attempt to merge questions into one *sequence* (markers in a single
row) was rejected: the encoder cost scales with tokens, not passes, so it was
only ~1.24× faster and much less accurate because markers ended up far from the
state. The official batch-dimension approach does not have that problem.

### Execution backends: WebGPU first, WASM second

`ProviderPlan` (`src/vendor/laya-ts/providers.ts`) is the whole of the backend
choice, and `webGpuPlan` is always tried first:

| | WebGPU plan | WASM plan |
| --- | --- | --- |
| `encoderProviders` | `["webgpu"]` | `["wasm"]` |
| `headProviders` | `["webgpu", "wasm"]` | `["wasm"]` |
| encoder `graphOptimizationLevel` | `disabled` | `basic` |
| tweets per pass | 8 | 1 |

Three things can move a run to the WASM plan, all decided by the pure
`decideWasmFallback` / `nextBackend`:

1. **Preflight.** `checkEncoderWebGpu` (adapter + `shader-f16`) refuses the GPU.
   Checked before the weights are pulled, so a browser with no WebGPU does not
   download ~700 MB before failing.
2. **Session.** `Agent.load` cannot create the WebGPU sessions. The WASM plan is
   then built from the same model URLs; nothing is re-downloaded when the service
   worker already has the files in Cache Storage (`assetIsCached`).
3. **Runtime.** A batch fails after the batch was already halved to a single
   tweet — device loss, an out-of-memory allocation, a WebGPU validation error,
   or a pass the output guards rejected. WASM is numerically the safe path, so it
   is worth one more attempt.

**Why the analysis restarts instead of resuming.** The two backends do not have
to agree on the last few percent of a borderline tweet, and a half-WebGPU /
half-WASM aggregate would be presented as one estimate. The worker posts
`restart` *before* rebuilding: the client drops every item it has received, the
count rewinds to `0 / total`, and only the WASM run can complete. The user sees
the count go back rather than a result that quietly changed halfway.

**Why the WASM plan is not simply "the same plan without WebGPU".** The
`disabled` optimization level exists because ORT's `SkipLayerNormalization`
fusion has a WebGPU kernel that rejects this graph — a problem the CPU kernels do
not have, and the fusions in `basic` are what make an fp16 encoder bearable on a
CPU. A wide batch buys GPU occupancy and only raises peak memory on the CPU, so
the fallback runs one tweet per pass. `numThreads` stays at 1 unless the page is
cross-origin isolated, because more WASM threads need `SharedArrayBuffer` and ORT
throws instead of dropping to one thread.

The cost is real: the CPU path is one to two orders of magnitude slower, and the
616 MB fp16 encoder has to fit in the WASM heap alongside the model bytes. The
status line therefore names the active backend and suggests lowering the sample
size rather than letting a 500-tweet run grind for hours.

### Asset caching

A Service Worker (`public/sw.js`) intercepts `/models/laya/**` and `/ort/**` and
serves them cache-first from Cache Storage, so the ~694 MB model is downloaded
once. The worker also streams `encoder.onnx` / `head.onnx` for a progress bar.
This requires a secure context, so it works on GitHub Pages and `localhost` but
not on a plain LAN IP.

### Model assets and hosting

The pinned checkpoint (`convaiinnovations/laya`, subfolder `multilingual`) is
exported in CI by `scripts/export_laya_ts.py` (from `NandhaKishorM/laya`,
Apache-2.0) into `public/models/laya` and cached between runs. Vite copies it
into `dist`. The encoder is fp16 with an fp32 output; the head stays fp32.
`graphOptimizationLevel` is disabled for the WebGPU encoder session to avoid an
ORT fusion (`SkipLayerNormalization`) whose WebGPU kernel rejects this graph; the
WASM encoder session uses `basic` instead, since that fusion is a GPU-only
problem.

### Size budget

Model ≈ 694 MB (encoder 616 MB + head 60 MB + tokenizer 34 MB), ONNX Runtime
(asyncify + jsep) ≈ 55 MB, app ≈ 0.5 MB. The plugin removes the duplicate Wasm
asset Vite would otherwise emit, and `scripts/validate-pages.mjs` fails the
build above 1 GB.

## Risks

| Risk | Mitigation |
| --- | --- |
| 694 MB first load | progress UI, Cache Storage reuse, clear messaging |
| Slow inference on weak GPUs | auto batch sizing, live ETA + cancel, sample cap |
| Huge archives | streaming unzip, lightweight records, per-entry reads |
| Model/site over budget | checksum + size gate in `build:pages` |
| No WebGPU, or a GPU that fails mid-run | WASM fallback + restart, active backend in the status line |
| The CPU path is 10–100× slower | one tweet per pass, `basic` fusions, WASM threads when cross-origin isolated, and a note to lower the sample size |
| The fp16 encoder may not fit in the WASM heap | the failure is reported as such, with a smaller sample size as the way out |
