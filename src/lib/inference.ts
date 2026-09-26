import type { AxisAnswer } from "./types.js";

/**
 * Which execution provider runs the model. Mirrors `EncoderBackend` in
 * `src/vendor/laya-ts/providers.ts`, which the worker reports verbatim.
 */
export type EncoderBackend = "webgpu" | "wasm";

export interface LoadProgressEvent {
  file: string;
  loaded: number;
  total: number;
  /** The bytes came from Cache Storage, so nothing was downloaded. */
  cached?: boolean;
}

export interface BackendInfo {
  backend: EncoderBackend;
  /** Why the active backend is not WebGPU; absent when it is. */
  reason?: string;
}

export interface AnalyzeProgress {
  done: number;
  total: number;
  elapsedMs: number;
  etaMs: number;
  batch: number;
  /** Backend that produced this count; absent only before the first pass. */
  backend?: EncoderBackend;
}

export interface AnalyzedItem {
  id: string;
  answers: Record<string, AxisAnswer>;
}

/** A run that switched backends and is starting again from the first tweet. */
export interface RestartInfo {
  backend: EncoderBackend;
  reason: string;
  total: number;
}

/**
 * Messages the inference worker sends. Mirrored by the `Response` type in
 * `src/workers/inference.worker.ts`, which imports this union so the two halves
 * cannot drift apart.
 */
export type WorkerResponse =
  | ({ type: "loadProgress" } & LoadProgressEvent)
  | ({ type: "ready" } & BackendInfo)
  | ({ type: "progress"; runId: number } & AnalyzeProgress)
  | { type: "item"; runId: number; item: AnalyzedItem }
  | ({ type: "restart"; runId: number } & RestartInfo)
  | { type: "done"; runId: number; total: number; backend: EncoderBackend }
  | { type: "error"; message: string; runId?: number }
  | { type: "cancelled"; runId: number };

interface PendingRun {
  items: AnalyzedItem[];
  resolve: (items: AnalyzedItem[]) => void;
  reject: (error: Error) => void;
  onProgress: (progress: AnalyzeProgress) => void;
  onItem: (item: AnalyzedItem) => void;
  onRestart: ((info: RestartInfo) => void) | null;
}

/** The slice of `Worker` this client uses; lets the tests drive the protocol. */
export interface WorkerLike {
  postMessage(message: unknown): void;
  terminate(): void;
  onmessage: ((event: MessageEvent<WorkerResponse>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
}

export type WorkerFactory = () => WorkerLike;

const defaultWorkerFactory: WorkerFactory = () =>
  new Worker(new URL("../workers/inference.worker.ts", import.meta.url), {
    type: "module",
  });

/** Main-thread client around the single inference worker. */
export class InferenceClient {
  private readonly worker: WorkerLike;
  private runCounter = 0;
  private pending = new Map<number, PendingRun>();
  private loadProgress: ((event: LoadProgressEvent) => void) | null = null;
  private ready: ((info: BackendInfo) => void) | null = null;
  private loadReject: ((error: Error) => void) | null = null;

  constructor(factory: WorkerFactory = defaultWorkerFactory) {
    const worker = factory();
    this.worker = worker;
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      this.handle(event.data);
    };
    worker.onerror = (event: ErrorEvent) => {
      const error = new Error(
        event.message || "推論ワーカーが停止しました。診断をリトライしてください。",
      );
      this.loadReject?.(error);
      for (const run of this.pending.values()) run.reject(error);
      this.pending.clear();
    };
  }

  private handle(data: WorkerResponse): void {
    switch (data.type) {
      case "loadProgress":
        this.loadProgress?.(data);
        break;
      case "ready":
        this.ready?.({ backend: data.backend, reason: data.reason });
        break;
      case "progress": {
        this.pending.get(data.runId)?.onProgress(data);
        break;
      }
      case "item": {
        const run = this.pending.get(data.runId);
        if (run) {
          run.items.push(data.item);
          run.onItem(data.item);
        }
        break;
      }
      case "restart": {
        // The worker is starting this run again on another backend, so every item
        // it has sent so far is void: keeping them would add up a partial WebGPU
        // result to the CPU one and look like a single complete estimate.
        const run = this.pending.get(data.runId);
        if (run) {
          run.items.length = 0;
          run.onRestart?.({ backend: data.backend, reason: data.reason, total: data.total });
        }
        break;
      }
      case "done": {
        const run = this.pending.get(data.runId);
        this.pending.delete(data.runId);
        run?.resolve(run.items);
        break;
      }
      case "cancelled": {
        const run = this.pending.get(data.runId);
        this.pending.delete(data.runId);
        run?.reject(new Error("解析をキャンセルしました"));
        break;
      }
      case "error": {
        const error = new Error(data.message);
        if (data.runId !== undefined) {
          const run = this.pending.get(data.runId);
          this.pending.delete(data.runId);
          run?.reject(error);
        } else {
          this.loadReject?.(error);
        }
        break;
      }
      default: {
        // Keeps this switch exhaustive: a message the worker can send but the
        // client ignores would otherwise go unhandled at runtime.
        const unhandled: never = data;
        void unhandled;
      }
    }
  }

  load(
    modelUrl: string,
    wasmPaths: string,
    onProgress: (event: LoadProgressEvent) => void,
  ): Promise<BackendInfo> {
    this.loadProgress = onProgress;
    return new Promise<BackendInfo>((resolve, reject) => {
      this.ready = resolve;
      this.loadReject = reject;
      this.worker.postMessage({ type: "load", modelUrl, wasmPaths });
    });
  }

  analyze(
    tweets: { id: string; text: string }[],
    handlers: {
      onProgress: (progress: AnalyzeProgress) => void;
      onItem: (item: AnalyzedItem) => void;
      /** Called when the run restarts on another backend; no items are kept. */
      onRestart?: (info: RestartInfo) => void;
    },
  ): Promise<AnalyzedItem[]> {
    const runId = ++this.runCounter;
    return new Promise<AnalyzedItem[]>((resolve, reject) => {
      this.pending.set(runId, {
        items: [],
        resolve,
        reject,
        onProgress: handlers.onProgress,
        onItem: handlers.onItem,
        onRestart: handlers.onRestart ?? null,
      });
      this.worker.postMessage({ type: "analyze", runId, tweets });
    });
  }

  cancel(): void {
    this.worker.postMessage({ type: "cancel" });
  }

  dispose(): void {
    this.worker.terminate();
  }
}
