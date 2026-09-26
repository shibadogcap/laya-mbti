/// <reference lib="webworker" />
import { Agent, type QuestionDef, type SystemOneResult } from "../vendor/laya-ts/agent.js";
import {
  assetIsCached,
  checkEncoderWebGpu,
  decideWasmFallback,
  errorText,
  isBatchSizeError,
  isNumericalError,
  isModelOutputError,
  nextBackend,
  wasmPlan,
  wasmThreadCount,
  webGpuPlan,
  type EncoderBackend,
  type ProviderPlan,
} from "../vendor/laya-ts/providers.js";
import { AXES, buildQuestions, randomizedQuestionPlan } from "../lib/mbti.js";
import { isFiniteAnswer } from "../lib/finite.js";
import type { WorkerResponse } from "../lib/inference.js";
import type { AxisAnswer } from "../lib/types.js";

type Request =
  | { type: "load"; modelUrl: string; wasmPaths: string }
  | { type: "analyze"; runId: number; tweets: { id: string; text: string }[] }
  | { type: "cancel" };

/**
 * The client owns the other half of this protocol. Importing the type keeps the
 * mirror honest: a message the client cannot handle fails to compile here.
 */
type Response = WorkerResponse;

const ctx = self as unknown as DedicatedWorkerGlobalScope;
const qCount = AXES.length;
/** The exported graph's dynamic batch dimension is capped at 64 rows. */
const MAX_ROWS = 64;

/** Unbiased draw for the per-run question shuffle; `Math.random` is enough. */
const cryptoRandom = (): number => Math.random();

/**
 * Wall-clock budget for a single pass. A phone GPU is far slower than a desktop
 * one, so the budget scales with the batch, but it stays bounded: a pass that
 * blows through it is treated as a hang rather than waited on forever.
 */
function passBudgetMs(itemCount: number): number {
  return Math.min(300_000, 30_000 + 20_000 * Math.max(1, itemCount));
}

/**
 * Rejects when the pass outlives its budget.
 *
 * When the GPU process dies mid-run, ORT can leave the promise pending instead
 * of rejecting, so the loop would never reach the batch-halving or WASM
 * fallback branches. The message is written to match the batch-size classifier
 * so a hang is handled exactly like a device loss.
 */
function withPassTimeout<T>(work: Promise<T>, itemCount: number): Promise<T> {
  const budget = passBudgetMs(itemCount);
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(
        new Error(
          `GPU が${Math.round(budget / 1000)}秒応答がありませんでした（device lost / ハング）`,
        ),
      );
    }, budget);
    work.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error: unknown) => {
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}
const MODEL_FILES = ["encoder.onnx", "head.onnx"] as const;
/** A lost WebGPU device can leave ORT's `release()` waiting; do not stall on it. */
const RELEASE_TIMEOUT_MS = 5000;

interface LoadContext {
  modelUrl: string;
  wasmPaths: string;
  /** WASM threads the CPU sessions may use; 1 unless the page is cross-origin isolated. */
  threads: number;
}

let agent: Agent | null = null;
let loadContext: LoadContext | null = null;
let activePlan: ProviderPlan = webGpuPlan();
let cancelled = false;
/** Bumped per run so a superseded analysis stops instead of racing a newer one. */
let runToken = 0;

function post(message: Response): void {
  ctx.postMessage(message);
}

/** Streams a model file once so the UI can show progress and caches warm up. */
async function warm(url: string, file: string): Promise<void> {
  const response = await fetch(url);
  if (!response.ok || !response.body) {
    throw new Error(`モデルの取得に失敗しました: ${url} (${response.status})`);
  }
  const total = Number(response.headers.get("content-length")) || 0;
  const reader = response.body.getReader();
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    loaded += value.byteLength;
    post({ type: "loadProgress", file, loaded, total });
  }
  if (total > 0 && loaded < total) {
    throw new Error(`モデルが途中で切れました。通信状況を確認してリトライしてください: ${file}`);
  }
}

/** Builds one provider on `plan`, reusing cached model files when there are any. */
async function createAgent(plan: ProviderPlan, ctxInfo: LoadContext): Promise<Agent> {
  const base = ctxInfo.modelUrl.endsWith("/") ? ctxInfo.modelUrl : `${ctxInfo.modelUrl}/`;
  for (const file of MODEL_FILES) {
    const url = `${base}${file}`;
    // The service worker stores the model on the first fetch, so rebuilding a
    // provider — which is what the fallback does — re-reads these from Cache
    // Storage instead of the network, and streaming them again would only
    // duplicate the download.
    if (await assetIsCached(url)) {
      post({ type: "loadProgress", file, loaded: 1, total: 1, cached: true });
      continue;
    }
    await warm(url, file);
  }
  return Agent.load(base, {
    wasmPaths: ctxInfo.wasmPaths,
    numThreads: plan.numThreads,
    encoderProviders: plan.encoderProviders,
    headProviders: plan.headProviders,
    encoderGraphOptimizationLevel: plan.encoderGraphOptimizationLevel,
    headGraphOptimizationLevel: plan.headGraphOptimizationLevel,
  });
}

/**
 * One short pass to prove the provider returns usable numbers.
 *
 * The encoder runs in fp16, and a phone GPU can overflow an activation to inf,
 * which reaches the head as NaN. That shows up as a run that dies a few hundred
 * posts in, after the wait for the model and a long GPU pass, so it is checked
 * once up front instead: a device that cannot produce a finite answer is handed
 * to the CPU before the user starts.
 */
const NUMERIC_PROBE_TEXT =
  "この文章は数値検証用の短いサンプルです。 yakinpaku na kotoba de arimasu.";

/** Returns the offending value's description, or null when the pass is usable. */
async function findNonFiniteAnswer(plan: ProviderPlan, ctxInfo: LoadContext): Promise<string | null> {
  const probe = await createAgent(plan, ctxInfo);
  try {
    const plan2 = randomizedQuestionPlan(() => 0.5);
    const results = await withPassTimeout(
      probe.predictBatch([NUMERIC_PROBE_TEXT], buildQuestions(plan2.order, plan2.flipOptions), {
        batchSize: AXES.length,
      }),
      1,
    );
    return findNonFinite(results);
  } finally {
    await releaseAgent(probe);
  }
}

function findNonFinite(results: SystemOneResult[]): string | null {
  for (const result of results) {
    for (const [id, answer] of Object.entries(result.answers)) {
      const bad = isFiniteAnswer(answer);
      if (bad) return `${id}: ${bad}`;
    }
  }
  return null;
}

/** Reads the model, preferring WebGPU and falling back to WASM if that cannot run. */
async function ensureAgent(ctxInfo: LoadContext): Promise<{ reason?: string }> {
  // The encoder is fp16, so the GPU has to expose shader-f16. Checked before
  // pulling the weights: a browser with no WebGPU would otherwise download
  // ~700 MB before failing, and the CPU path does not care about the GPU at all.
  const capability = await checkEncoderWebGpu();
  const preferred = webGpuPlan(ctxInfo.threads);
  const first = capability.ok ? preferred : wasmPlan(ctxInfo.threads);
  const reasons: string[] = capability.ok ? [] : [capability.message];
  try {
    agent = await createAgent(first, ctxInfo);
    activePlan = first;
    if (first.backend === "webgpu") {
      const broken = await findNonFiniteAnswer(first, ctxInfo);
      if (broken) {
        throw new Error(
          `この GPU は fp16 の計算が安定せず、検証用の1件で不正な値（${broken}）が出ました。` +
            `WebGPU は使わず、CPU（WASM）で解析します。`,
        );
      }
    }
  } catch (error) {
    const decision = decideWasmFallback({
      error,
      phase: capability.ok ? "session" : "preflight",
    });
    if (nextBackend(first.backend, decision) === null) throw error;
    const fallback = wasmPlan(ctxInfo.threads);
    activePlan = fallback;
    agent = await createAgent(fallback, ctxInfo);
    reasons.push(decision.reason);
  }
  return reasons.length > 0 ? { reason: reasons.join(" ") } : {};
}

function releaseAgent(current: Agent | null): Promise<void> {
  if (!current) return Promise.resolve();
  // The GPU memory goes away with the device, so a release that never settles
  // must not hold up the CPU rebuild that follows it.
  return Promise.race([
    current.dispose().catch(() => undefined),
    new Promise<void>((resolve) => setTimeout(resolve, RELEASE_TIMEOUT_MS)),
  ]);
}

/** Replaces a dead WebGPU provider with a WASM one, freeing the GPU session first. */
async function switchToWasm(
  runId: number,
  total: number,
  reason: string,
): Promise<{ plan: ProviderPlan; agent: Agent }> {
  if (!loadContext) throw new Error(reason);
  const previous = agent;
  // Stop handing the dead provider out before it is released.
  agent = null;
  const fallback = wasmPlan(loadContext.threads);
  activePlan = fallback;
  // Announced before the rebuild: the count starts again from zero, and the
  // WebGPU items already sent are dropped, so a partial result is never
  // presented next to the CPU one as if the two belonged together.
  post({ type: "restart", runId, backend: fallback.backend, reason, total });
  await releaseAgent(previous);
  let replacement: Agent;
  try {
    replacement = await createAgent(fallback, loadContext);
  } catch (error) {
    throw new Error(
      `WebGPU でも CPU（WASM）でも推論を開始できませんでした。${errorText(error)}`,
    );
  }
  agent = replacement;
  return { plan: fallback, agent: replacement };
}

/** Tweets per pass, clamped to the 64-row limit of the exported graphs. */
function statesPerChunkFor(plan: ProviderPlan): number {
  return Math.max(1, Math.min(plan.statesPerChunk, Math.floor(MAX_ROWS / qCount)));
}

/** The final message when neither the GPU nor the CPU can finish a single tweet. */
function fatalAnalyzeError(error: unknown, plan: ProviderPlan): Error {
  if (plan.backend === "wasm") {
    return new Error(
      `1 件の投稿ずつに分けても CPU（WASM）で推論できませんでした（メモリ不足の可能性があります）。` +
        `解析する最大件数を絞ってからリトライしてください。 (${errorText(error)})`,
    );
  }
  return new Error(
    `1 件の投稿ずつに分けても WebGPU で推論できませんでした（GPU のメモリ不足、またはデバイスのエラーです）。` +
      `GPU を使う他のタブやアプリを閉じてからリトライしてください。 (${errorText(error)})`,
  );
}

async function analyze(
  runId: number,
  tweets: { id: string; text: string }[],
): Promise<void> {
  const token = ++runToken;
  const superseded = () => token !== runToken;
  const current = agent;
  if (!current) {
    post({ type: "error", runId, message: "モデルが読み込まれていません" });
    return;
  }
  cancelled = false;
  // One question layout per run: the axis order and which pole is listed first
  // are shuffled so a fixed "E or I, E first" prompt cannot bake a position
  // bias into every run of the same archive. Every post in this run shares the
  // layout, so the four axes stay comparable inside one aggregate.
  const questionPlan = randomizedQuestionPlan(cryptoRandom);
  const questions: Record<string, QuestionDef> = buildQuestions(
    questionPlan.order,
    questionPlan.flipOptions,
  );
  // Chunks are padded to a fixed row count, so grouping by text length buys no
  // throughput. The payload order is kept instead, which is chronological: the
  // reading feed then walks the archive the same way the user reads it.
  const ordered = tweets;
  let active = current;
  let plan = activePlan;
  let started = performance.now();
  let done = 0;
  let statesPerChunk = statesPerChunkFor(plan);

  for (;;) {
    while (done < ordered.length) {
      if (cancelled || superseded()) {
        post({ type: "cancelled", runId });
        return;
      }
      const chunk = ordered.slice(done, done + statesPerChunk);
      let results;
      try {
        results = await withPassTimeout(
          active.predictBatch(chunk.map((tweet) => tweet.text), questions, {
            batchSize: statesPerChunk * qCount,
          }),
          chunk.length,
        );
      } catch (error) {
        if (cancelled || superseded()) {
          post({ type: "cancelled", runId });
          return;
        }
        // Out of memory, WebGPU limit/validation failures and device loss all clear
        // on a smaller batch, so halve it and retry until a single tweet per pass.
        // A non-finite result is excluded: a GPU whose fp16 arithmetic overflows
        // returns NaN at every batch size, so retrying smaller only burns three
        // passes before the CPU fallback that actually helps.
        if (
          statesPerChunk > 1 &&
          (isBatchSizeError(error) || isModelOutputError(error)) &&
          !isNumericalError(error)
        ) {
          statesPerChunk = Math.max(1, Math.floor(statesPerChunk / 2));
          continue;
        }
        const decision = decideWasmFallback({ error, phase: "runtime" });
        const backend: EncoderBackend | null = nextBackend(plan.backend, decision);
        if (backend === null) {
          // Nothing to fall back to: either a CPU failure (final) or an error the
          // CPU would raise identically, which is reported as it came.
          if (!decision.fallback) throw error;
          throw fatalAnalyzeError(error, plan);
        }
        // No second chance on the GPU: rebuild on the CPU and start over.
        const switched = await switchToWasm(runId, ordered.length, decision.reason);
        plan = switched.plan;
        active = switched.agent;
        statesPerChunk = statesPerChunkFor(plan);
        started = performance.now();
        done = 0;
        break;
      }

      results.forEach((result, index) => {
        post({
          type: "item",
          runId,
          item: {
            id: chunk[index].id,
            answers: result.answers as Record<string, AxisAnswer>,
          },
        });
      });

      done += chunk.length;
      const elapsedMs = performance.now() - started;
      post({
        type: "progress",
        runId,
        done,
        total: ordered.length,
        elapsedMs,
        etaMs: done > 0 ? (elapsedMs / done) * (ordered.length - done) : 0,
        batch: statesPerChunk,
        backend: plan.backend,
      });
    }
    if (done >= ordered.length) {
      post({ type: "done", runId, total: ordered.length, backend: plan.backend });
      return;
    }
  }
}

ctx.onmessage = (event: MessageEvent<Request>) => {
  const data = event.data;
  if (data.type === "cancel") {
    cancelled = true;
    return;
  }
  if (data.type === "load") {
    const ctxInfo: LoadContext = {
      modelUrl: data.modelUrl,
      wasmPaths: data.wasmPaths,
      threads: wasmThreadCount({
        crossOriginIsolated: ctx.crossOriginIsolated,
        hardwareConcurrency: navigator.hardwareConcurrency,
      }),
    };
    loadContext = ctxInfo;
    ensureAgent(ctxInfo)
      .then((info) => post({ type: "ready", backend: activePlan.backend, reason: info.reason }))
      .catch(async (error: unknown) => {
        await releaseAgent(agent);
        agent = null;
        post({
          type: "error",
          message: error instanceof Error ? error.message : String(error),
        });
      });
    return;
  }
  if (data.type === "analyze") {
    analyze(data.runId, data.tweets).catch((error: unknown) =>
      post({
        type: "error",
        runId: data.runId,
        message: error instanceof Error ? error.message : String(error),
      }),
    );
  }
};
