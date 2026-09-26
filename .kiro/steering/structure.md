# Structure Steering — laya-mbti

```
src/
├─ index.tsx                 # Solid entrypoint
├─ App.tsx                   # State machine and page composition
├─ components/               # Presentational Solid components
│  ├─ DropZone.tsx
│  ├─ PeriodPicker.tsx
│  └─ ResultCard.tsx
├─ workers/                  # Web Workers (never imported by the main thread)
│  ├─ archive.worker.ts      # zip.js streaming + archive parsing
│  └─ inference.worker.ts    # laya-ts Agent load + batched predict
├─ vendor/laya-ts/           # Vendored official Laya TS runtime (Apache-2.0)
│  ├─ agent.ts               # + predictBatch / _answerFor (modified)
│  ├─ providers.ts           # browser-only WebGPU/WASM provider (modified)
│  ├─ common.ts              # sequence building, softmax, collate
│  ├─ tokenizer.ts           # HF tokenizer.json BPE
│  ├─ hooks.ts               # hook lifecycle
│  └─ structured.ts          # JSON-schema decisions (unused by the app)
└─ lib/
   ├─ types.ts               # Shared domain types
   ├─ parse.ts               # Pure X archive parsers (js/csv/account)
   ├─ archive.ts             # Main-thread client for archive.worker
   ├─ inference.ts           # Main-thread client for inference.worker
   ├─ mbti.ts                # Axis definitions + Laya question batch
   ├─ filter.ts              # Pre-filtering (length, URLs, replies, period)
   ├─ aggregate.ts           # Margin-weighted axis aggregation
   └─ format.ts              # Byte/duration/date formatting
scripts/
├─ export_laya_ts.py         # fp16 split-ONNX export (from NandhaKishorM/laya)
└─ validate-pages.mjs        # Deploy integrity + 1 GB budget gate
.kiro/                       # Steering, specs, hooks, agents, skills, powers
public/                      # Vite static root (models generated here in CI)
```

## Conventions

- `lib/parse.ts`, `lib/filter.ts`, and `lib/aggregate.ts` are pure and must stay
  unit-testable without a browser.
- `src/vendor/laya-ts/` is vendored upstream code: keep changes minimal, mark
  them in the file header, and record them in `NOTICE`.
- Worker message types live next to the worker that owns them and are mirrored
  in the corresponding `lib/*.ts` client. Keep both sides in sync.
- Axis and question definitions are the single source of truth in `lib/mbti.ts`;
  never hardcode axis letters elsewhere.
- UI copy is Japanese. Code, identifiers, and comments are English.
