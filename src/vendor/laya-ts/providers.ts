/** ONNX Runtime Web session shim for laya-ts.
 *
 * Vendored from NandhaKishorM/laya `laya-ts/src/providers.ts` (Apache-2.0) and
 * modified for laya-mbti:
 *   - browser-only (the Node provider was removed)
 *   - ONNX Runtime is bundled statically instead of a `@vite-ignore` dynamic import
 *   - the encoder prefers WebGPU and falls back to WASM; the head may use WASM
 *   - the encoder→head handoff uses a flat `Float32Array` instead of nested arrays,
 *     which otherwise dominates cost for batched inputs
 *   - sessions are created from a URL so ORT streams the ~600 MB model
 *   - a WebGPU capability preflight (fp16 / `shader-f16`) before the encoder session
 *   - OOM/batch-size error classification is exported for the worker's retry loop
 *   - `ProviderPlan` / `decideWasmFallback` describe the WebGPU-first, WASM-second
 *     encoder choice so the worker can rebuild and restart without duplicating policy
 */
import * as ort from "onnxruntime-web";

export interface Batch {
  inputIds: number[][];
  attentionMask: number[][];
  markerPos: number[][];
  markerMask: boolean[][];
  qtype: number[];
}

/** Encoder output kept flat; the head consumes it without a nested-array round trip. */
export interface EncodedHidden {
  data: Float32Array;
  dims: number[];
}

export interface SessionProvider {
  runEncoder(batch: Batch): Promise<{ lastHidden: EncodedHidden }>;
  runHead(
    hidden: EncodedHidden,
    batch: Batch,
  ): Promise<{ logits: number[][]; act: number[][] }>;
  /**
   * Releases both sessions. The WebGPU fallback builds a second provider while
   * the first still holds the encoder in GPU memory, so the old provider has to
   * be freed before the new one is created. Optional: a stub provider may omit it.
   */
  dispose?(): Promise<void>;
}

function toNested(data: ArrayLike<number | bigint | boolean>, dims: number[]): any {
  const flat = Array.from(data as any, (v: any) => (typeof v === "bigint" ? Number(v) : v));
  if (dims.length === 0) return flat[0];
  const rec = (d: number, off: number): any => {
    if (d === dims.length - 1) return flat.slice(off, off + dims[d]);
    const step = dims.slice(d + 1).reduce((a, b) => a * b, 1);
    const out: any[] = [];
    for (let i = 0; i < dims[d]; i++) out.push(rec(d + 1, off + i * step));
    return out;
  };
  return rec(0, 0);
}

/** Encoder feeds: input_ids + attention_mask (int64). */
export function feed(b: Batch): Record<string, ort.Tensor> {
  const n = b.inputIds.length;
  const L = Math.max(1, ...b.inputIds.map((r) => r.length));
  const inputIds = new BigInt64Array(n * L);
  const attentionMask = new BigInt64Array(n * L);
  b.inputIds.forEach((row, rowIndex) => {
    for (let columnIndex = 0; columnIndex < L; columnIndex++) {
      const offset = rowIndex * L + columnIndex;
      inputIds[offset] = BigInt(row[columnIndex] ?? 0);
      attentionMask[offset] = BigInt(b.attentionMask[rowIndex]?.[columnIndex] ?? 0);
    }
  });
  return {
    input_ids: new ort.Tensor("int64", inputIds, [n, L]),
    attention_mask: new ort.Tensor("int64", attentionMask, [n, L]),
  };
}

/** Head feeds: encoder hidden + marker_pos/mask + qtype + padding mask. */
export function feedHead(
  hidden: EncodedHidden,
  b: Batch,
): Record<string, ort.Tensor> {
  const n = b.markerPos.length;
  const k = Math.max(1, ...b.markerPos.map((r) => r.length));
  const S = hidden.dims[1] ?? 1;
  const markerPos = new BigInt64Array(n * k);
  const markerMask = new Uint8Array(n * k);
  const qtype = new BigInt64Array(n);
  const attentionMask = new BigInt64Array(n * S);
  b.markerPos.forEach((row, rowIndex) => {
    for (let columnIndex = 0; columnIndex < k; columnIndex++) {
      markerPos[rowIndex * k + columnIndex] = BigInt(row[columnIndex] ?? 0);
      markerMask[rowIndex * k + columnIndex] = b.markerMask[rowIndex]?.[columnIndex] ? 1 : 0;
    }
  });
  b.attentionMask.forEach((row, rowIndex) => {
    for (let columnIndex = 0; columnIndex < S; columnIndex++) {
      attentionMask[rowIndex * S + columnIndex] = BigInt(row[columnIndex] ?? 0);
    }
    qtype[rowIndex] = BigInt(b.qtype[rowIndex] ?? 0);
  });
  return {
    hidden_states: new ort.Tensor("float32", hidden.data, hidden.dims),
    marker_pos: new ort.Tensor("int64", markerPos, [n, k]),
    marker_mask: new ort.Tensor("bool", markerMask, [n, k]),
    qtype: new ort.Tensor("int64", qtype, [n, 1]),
    attention_mask: new ort.Tensor("int64", attentionMask, [n, S]),
  };
}

function pickOutput(out: Record<string, any>, names: string[]): any {
  for (const n of names) if (out[n] !== undefined) return out[n];
  return Object.values(out)[0];
}

/** Flattens an error (message plus a distinguishing `name`, e.g. `GPUDeviceLostError`). */
export function errorText(error: unknown): string {
  const name = (error as { name?: unknown } | null)?.name;
  const message = (error as { message?: unknown } | null)?.message;
  const label = typeof name === "string" && name !== "" && name !== "Error" ? name : "";
  const body = message === undefined || message === null || message === "" ? String(error) : String(message);
  return label === "" ? body : `${label}: ${body}`;
}

/** Allocation failures, in any of the wordings ORT and the WebGPU stack use. */
const MEMORY_PATTERN =
  /out of memory|oom|memory|allocat|insufficient (?:memory|resources)|exceeds the (?:memory|buffer)/i;

/**
 * WebGPU rejections that a smaller batch can clear: device loss, buffer/binding
 * limits, WGSL and dispatch validation. A big batch trips these; a single item
 * usually does not, so the worker halves its batch and retries.
 */
const WEBGPU_LIMIT_PATTERN =
  /device lost|device was lost|lost the device|context lost|adapter|out of bounds|out-of-bounds|exceed|too large|maximum|limit|validation|invalid|not supported|unsupported|buffer size|buffer is too|bind ?group|pipeline|compilation|shader|dispatch/i;

/** True for allocation failures, i.e. errors worth annotating as out-of-memory. */
export function isMemoryError(error: unknown): boolean {
  return MEMORY_PATTERN.test(errorText(error));
}

/**
 * The messages `validateHeadOutput` / `validateAnswer` raise when a pass cannot be
 * decoded (missing rows, NaN, a distribution that does not add up). Exported so the
 * worker's retry and fallback logic recognises them without duplicating the wording.
 */
const MODEL_OUTPUT_PATTERN = /NaN|Infinity|有限|確率|出力|行|答え|集計|計算/i;

/** True when the model output itself was unusable, whatever the execution provider. */
export function isModelOutputError(error: unknown): boolean {
  return MODEL_OUTPUT_PATTERN.test(errorText(error));
}

/**
 * True when the failure may be caused by the batch being too large for the GPU.
 * The worker uses this to shrink `statesPerChunk` and try again; it stays true for
 * device-lost and validation messages, which is what ORT reports instead of OOM.
 */
export function isBatchSizeError(error: unknown): boolean {
  const text = errorText(error);
  return MEMORY_PATTERN.test(text) || WEBGPU_LIMIT_PATTERN.test(text);
}

/**
 * WebGPU features the fp16 encoder needs. The exported encoder runs its weights and
 * activations in fp16, so ORT has to request `shader-f16`; a GPU without it fails
 * deep inside shader compilation with an opaque message.
 */
export const ENCODER_REQUIRED_FEATURES: readonly string[] = ["shader-f16"];

interface FeatureSet {
  has(feature: string): boolean;
}

interface AdapterLike {
  features: FeatureSet;
}

interface GpuLike {
  requestAdapter(): Promise<AdapterLike | null>;
}

export type EncoderCapabilityReason =
  | "ok"
  | "no-webgpu"
  | "no-adapter"
  | "missing-feature";

export interface EncoderCapability {
  ok: boolean;
  reason: EncoderCapabilityReason;
  /** The required WebGPU feature the adapter lacks, when `reason` is `missing-feature`. */
  feature?: string;
  /**
   * User-facing Japanese explanation of why WebGPU cannot run the encoder, or
   * empty when it can. The CPU path is the worker's decision, so this text only
   * states the cause; `decideWasmFallback` adds what will be done about it.
   */
  message: string;
}

const CAPABILITY_MESSAGE: Record<Exclude<EncoderCapabilityReason, "ok" | "missing-feature">, string> = {
  "no-webgpu":
    "この環境では WebGPU を使えません。" +
    "WebGPU は Chrome または Edge の最新版で、https または localhost 経由でないと使えません。",
  "no-adapter":
    "WebGPU のアダプターを取得できませんでした。GPU がブロックリストに載っているか、ブラウザのハードウェアアクセラレーションが無効です。",
};

/**
 * Pure verdict on whether this GPU can run the fp16 encoder, from the pieces of
 * WebGPU the app can observe. `features` is `null` when no adapter was obtained.
 */
export function encoderCapability(input: {
  gpu: unknown;
  features?: FeatureSet | null;
}): EncoderCapability {
  if (!input.gpu) return { ok: false, reason: "no-webgpu", message: CAPABILITY_MESSAGE["no-webgpu"] };
  const features = input.features ?? null;
  if (!features) {
    return { ok: false, reason: "no-adapter", message: CAPABILITY_MESSAGE["no-adapter"] };
  }
  const missing = ENCODER_REQUIRED_FEATURES.find((feature) => !features.has(feature));
  if (missing) {
    return {
      ok: false,
      reason: "missing-feature",
      feature: missing,
      message:
        `この GPU は fp16 推論に対応していません（WebGPU の ${missing} 機能がありません）。` +
        "GPU ドライバーを更新するか、fp16 に対応した GPU で開いてください。",
    };
  }
  return { ok: true, reason: "ok", message: "" };
}

function readGpu(): GpuLike | null {
  const gpu = (globalThis as { navigator?: { gpu?: GpuLike } }).navigator?.gpu;
  return gpu && typeof gpu.requestAdapter === "function" ? gpu : null;
}

/**
 * Asks for an adapter and checks the fp16 feature, before any session is created.
 * Only the adapter is requested: ORT owns the device lifecycle, and asking for a
 * second device here would be thrown away (or would fail for reasons that are not
 * about fp16).
 */
export async function checkEncoderWebGpu(): Promise<EncoderCapability> {
  const gpu = readGpu();
  if (!gpu) return encoderCapability({ gpu: null });
  let adapter: AdapterLike | null = null;
  try {
    adapter = await gpu.requestAdapter();
  } catch {
    adapter = null;
  }
  return encoderCapability({ gpu, features: adapter?.features ?? null });
}

export type GraphOptimizationLevel =
  | "disabled"
  | "basic"
  | "extended"
  | "layout"
  | "all";

/** Which execution provider runs the encoder, and with it the whole model. */
export type EncoderBackend = "webgpu" | "wasm";

/**
 * One way to run the model: ORT session options plus the batching the worker
 * should use on it. WebGPU is the preferred encoder; WASM is the fallback, so
 * the two plans differ in every field that a different machine needs.
 */
export interface ProviderPlan {
  backend: EncoderBackend;
  encoderProviders: string[];
  headProviders: string[];
  encoderGraphOptimizationLevel: GraphOptimizationLevel;
  headGraphOptimizationLevel: GraphOptimizationLevel;
  /** WASM threads for the CPU sessions; see `wasmThreadCount`. */
  numThreads: number;
  /**
   * Tweets per encoder pass. A wide batch only buys GPU occupancy, while on the
   * CPU it multiplies peak memory, so the WASM plan runs one tweet at a time.
   */
  statesPerChunk: number;
}

/** Preferred plan: the fp16 encoder on WebGPU, with WASM still allowed for the head. */
export function webGpuPlan(numThreads = 1): ProviderPlan {
  return {
    backend: "webgpu",
    encoderProviders: ["webgpu"],
    headProviders: ["webgpu", "wasm"],
    // Disabled on purpose: ORT's SkipLayerNormalization fusion has a WebGPU
    // kernel that rejects this graph's Beta.
    encoderGraphOptimizationLevel: "disabled",
    headGraphOptimizationLevel: "basic",
    numThreads,
    statesPerChunk: 8,
  };
}

/**
 * Fallback plan: everything on the WASM execution provider, so no GPU device is
 * touched at all. The `disabled` optimization level above is a WebGPU
 * workaround; the CPU kernels do not have the problem it works around, and the
 * fusions in `basic` are what make an fp16 encoder bearable on a CPU.
 */
export function wasmPlan(numThreads = 1): ProviderPlan {
  return {
    backend: "wasm",
    encoderProviders: ["wasm"],
    headProviders: ["wasm"],
    encoderGraphOptimizationLevel: "basic",
    headGraphOptimizationLevel: "basic",
    numThreads,
    statesPerChunk: 1,
  };
}

/**
 * WASM threads the CPU sessions may use. More than one needs `SharedArrayBuffer`,
 * which needs COOP/COEP (`crossOriginIsolated`); without those headers ORT
 * throws instead of dropping to one thread, so the single-threaded count is not
 * a preference but a requirement of the host page.
 */
export function wasmThreadCount(input: {
  crossOriginIsolated?: boolean;
  hardwareConcurrency?: number;
}): number {
  if (!input.crossOriginIsolated) return 1;
  const cores = Number(input.hardwareConcurrency);
  if (!Number.isFinite(cores) || cores <= 1) return 1;
  return Math.max(1, Math.min(4, Math.floor(cores) - 1));
}

/** Where in an analysis the WebGPU path gave up. */
export type WebGpuFailurePhase =
  /** The fp16 capability preflight refused the GPU. No session was attempted. */
  | "preflight"
  /** A session could not be created (fp16 kernels, adapter, device request). */
  | "session"
  /** A pass failed while inference was already running. */
  | "runtime";

/** A model file that is missing rather than unusable: a fallback would fail the same way. */
const FETCH_PATTERN = /fetch failed|failed to fetch|network ?error|failed to load|\b(?:40[0134]|5\d\d)\b/i;

const FALLBACK_INTRO = "WebGPU で推論できなかったため、CPU（WASM）で最初から推論します。";

export interface WasmFallbackDecision {
  /** True when the worker should build a WASM provider and restart the analysis. */
  fallback: boolean;
  /** User-facing Japanese reason, empty when `fallback` is false. */
  reason: string;
}

/**
 * Pure decision on whether a WebGPU failure is worth a second run on the CPU.
 *
 * `preflight` and `session` failures are decided by the backend itself, so they
 * always fall back; a `runtime` failure has to be a backend failure, because a
 * broken question definition or a cancel would fail identically on the CPU. The
 * analysis is restarted from the first tweet either way, so a partial WebGPU
 * result is never mixed with the WASM one.
 */
export function decideWasmFallback(input: {
  error: unknown;
  phase: WebGpuFailurePhase;
}): WasmFallbackDecision {
  const { error, phase } = input;
  if (phase !== "runtime") {
    if (FETCH_PATTERN.test(errorText(error))) return { fallback: false, reason: "" };
    return { fallback: true, reason: `${FALLBACK_INTRO} (${errorText(error)})` };
  }
  if (!isBatchSizeError(error) && !isModelOutputError(error)) {
    return { fallback: false, reason: "" };
  }
  return { fallback: true, reason: `${FALLBACK_INTRO} (${errorText(error)})` };
}

/**
 * The backend to move to, or `null` to keep failing. WASM is the last resort:
 * once the CPU path fails there is no third backend, so its error is final.
 */
export function nextBackend(
  current: EncoderBackend,
  decision: WasmFallbackDecision,
): EncoderBackend | null {
  if (current !== "webgpu" || !decision.fallback) return null;
  return "wasm";
}

/**
 * True when the service worker has already stored an asset in Cache Storage, in
 * which case rebuilding a provider on the fallback path costs no download.
 */
export async function assetIsCached(url: string): Promise<boolean> {
  try {
    const store = (globalThis as { caches?: { match?: (u: string) => Promise<unknown> } }).caches;
    if (!store || typeof store.match !== "function") return false;
    return Boolean(await store.match(url));
  } catch {
    // A blocked or full cache is not an error: the fetch below just re-downloads.
    return false;
  }
}

export interface WebProviderOptions {
  /** Directory containing the ONNX Runtime `.wasm`/`.mjs` assets (self-hosted). */
  wasmPaths?: string;
  /** Override the encoder execution providers. Defaults to WebGPU only. */
  encoderProviders?: string[];
  /** Override the head execution providers. Defaults to WebGPU with WASM fallback. */
  headProviders?: string[];
  /** WASM threads for the WASM sessions; 1 avoids needing COOP/COEP headers. */
  numThreads?: number;
  /** Runtime graph optimization level for the encoder session. */
  encoderGraphOptimizationLevel?: GraphOptimizationLevel;
  /** Runtime graph optimization level for the head session. */
  headGraphOptimizationLevel?: GraphOptimizationLevel;
}

export interface WebBundle {
  dir: string;
  cfg: any;
  tokenizerJson: unknown | null;
}

/** Cache-first fetch through Cache Storage, with a network fallback. */
async function fetchArrayBuffer(url: string): Promise<ArrayBuffer> {
  const g = globalThis as unknown as { caches?: any };
  let cache: any = null;
  let hit: any = null;
  try {
    if (g.caches && typeof g.caches.open === "function") {
      cache = await g.caches.open("laya-ts");
      hit = await cache.match(url);
    }
  } catch {
    cache = null;
  }
  if (hit) {
    try {
      return await hit.arrayBuffer();
    } catch {
      /* fall through to network */
    }
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch failed for ${url}: ${res.status}`);
  if (cache) {
    try {
      await cache.put(url, res.clone());
    } catch {
      /* cache full/blocked: still return network bytes */
    }
  }
  return await res.arrayBuffer();
}

async function fetchJson(url: string): Promise<unknown> {
  const buf = await fetchArrayBuffer(url);
  return JSON.parse(new TextDecoder().decode(buf));
}

function baseUrlFor(repoOrUrl: string, subfolder?: string | null): string {
  const sub = subfolder ? `/${subfolder.replace(/^\/+|\/+$/g, "")}` : "";
  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(repoOrUrl)) {
    return `${repoOrUrl.replace(/\/+$/, "")}${sub}`;
  }
  return `https://huggingface.co/${repoOrUrl}/resolve/main${sub}`;
}

export async function loadWebBundle(
  repoOrUrl: string,
  opts?: { subfolder?: string | null },
): Promise<WebBundle> {
  const base = baseUrlFor(repoOrUrl, opts?.subfolder ?? null);
  let cfg: any;
  try {
    cfg = await fetchJson(`${base}/rl_agent_config.json`);
  } catch {
    throw new Error(
      `Incompatible model: ${JSON.stringify(repoOrUrl)} does not contain 'rl_agent_config.json'.`,
    );
  }
  let tokenizerJson: unknown | null = null;
  for (const candidate of ["tokenizer.json", "tokenizer/tokenizer.json"]) {
    try {
      tokenizerJson = await fetchJson(`${base}/${candidate}`);
      break;
    } catch {
      // Try the next supported layout.
    }
  }
  return { dir: base, cfg, tokenizerJson };
}

/** Japanese name for a provider list, for session-setup error messages. */
function backendLabel(executionProviders: string[]): string {
  if (executionProviders.includes("webgpu")) return "WebGPU";
  if (executionProviders.includes("wasm")) return "CPU（WASM）";
  return executionProviders.join("、") || "既定の実行プロバイダー";
}

export async function createWebProvider(
  modelUrl: string,
  opts?: WebProviderOptions,
): Promise<SessionProvider> {
  if (opts?.wasmPaths) ort.env.wasm.wasmPaths = opts.wasmPaths;
  if (opts?.numThreads && ort.env?.wasm) ort.env.wasm.numThreads = opts.numThreads;
  const base = modelUrl.replace(/\/+$/, "");

  const encoderOptimization =
    opts?.encoderGraphOptimizationLevel ?? "disabled";
  const headOptimization =
    opts?.headGraphOptimizationLevel ?? "basic";
  const encoderProviders = opts?.encoderProviders ?? ["webgpu"];
  const headProviders = opts?.headProviders ?? ["webgpu", "wasm"];
  // Checked before the session exists: an fp16 encoder on a GPU without
  // `shader-f16` fails during session setup. The CPU plan skips this entirely,
  // since it is not asking the GPU for anything.
  if (encoderProviders.includes("webgpu")) {
    const capability = await checkEncoderWebGpu();
    if (!capability.ok) throw new Error(capability.message);
  }
  // The encoder is ~616 MB: a half-built pair of sessions would hold it in
  // memory with nothing pointing at it, so a failure releases what did start.
  let enc: ort.InferenceSession | null = null;
  let head: ort.InferenceSession;
  try {
    enc = await ort.InferenceSession.create(`${base}/encoder.onnx`, {
      executionProviders: encoderProviders,
      // Keep portable ops instead of letting ORT fuse LayerNorm into
      // SkipLayerNormalization, whose WebGPU kernel rejects this graph's Beta.
      graphOptimizationLevel: encoderOptimization,
    });
    head = await ort.InferenceSession.create(`${base}/head.onnx`, {
      executionProviders: headProviders,
      graphOptimizationLevel: headOptimization,
    });
  } catch (error) {
    await enc?.release().catch(() => undefined);
    const where = enc === null ? "encoder" : "head";
    const label = backendLabel(where === "encoder" ? encoderProviders : headProviders);
    const hint =
      label === "WebGPU"
        ? "WebGPU 対応ブラウザ（Chrome/Edge）で開いてください。"
        : "ブラウザのメモリが不足している可能性があります。他のタブやアプリを閉じてください。";
    throw new Error(`${label} で ${where} を初期化できませんでした。${hint} (${errorText(error)})`);
  }
  const encoderSession = enc;
  const headSession = head;

  return {
    runEncoder: async (b) => {
      let out: Record<string, ort.Tensor>;
      try {
        out = await encoderSession.run(feed(b));
      } catch (e) {
        if (isMemoryError(e)) {
          throw new Error(`${errorText(e)} (${backendLabel(encoderProviders)} のメモリ不足)`);
        }
        throw e;
      }
      const t = pickOutput(out, ["last_hidden_state", "lastHidden", "hidden_states"]);
      const data =
        t.data instanceof Float32Array
          ? t.data
          : Float32Array.from(t.data as ArrayLike<number>, (v) => Number(v));
      return { lastHidden: { data, dims: t.dims } };
    },
    runHead: async (hidden, b) => {
      let out: Record<string, ort.Tensor>;
      try {
        out = await headSession.run(feedHead(hidden, b));
      } catch (e) {
        if (isMemoryError(e)) {
          const onGpu = headProviders.includes("webgpu");
          throw new Error(
            onGpu
              ? `${errorText(e)} (メモリ不足: バッチを小さくしてください)`
              : `${errorText(e)} (CPU でのメモリ不足: 解析する件数を減らしてください)`,
          );
        }
        throw e;
      }
      const vals = Object.values(out);
      const lt = pickOutput(out, ["logits"]);
      const at = pickOutput(out, ["act_logits", "act"]) ?? vals[1] ?? vals[0];
      return { logits: toNested(lt.data, lt.dims), act: toNested(at.data, at.dims) };
    },
    dispose: async () => {
      // Released independently: a stuck WebGPU device must not keep the head
      // session (or the reverse) alive, since the fallback needs the memory.
      await Promise.allSettled([encoderSession.release(), headSession.release()]);
    },
  };
}
