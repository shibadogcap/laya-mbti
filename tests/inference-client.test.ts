import { describe, expect, it } from "vitest";
import {
  InferenceClient,
  type RestartInfo,
  type WorkerLike,
  type WorkerResponse,
} from "../src/lib/inference.js";

/** Stands in for the worker so a test can replay the message sequence. */
class FakeWorker implements WorkerLike {
  onmessage: ((event: MessageEvent<WorkerResponse>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  readonly sent: unknown[] = [];

  postMessage(message: unknown): void {
    this.sent.push(message);
  }

  terminate(): void {}

  deliver(data: WorkerResponse): void {
    this.onmessage?.({ data } as MessageEvent<WorkerResponse>);
  }

  /** The runId of the single `analyze` request, or -1. */
  get runId(): number {
    const request = this.sent.find(
      (message) => (message as { type?: string }).type === "analyze",
    ) as { runId: number } | undefined;
    return request?.runId ?? -1;
  }
}

const client = () => {
  const worker = new FakeWorker();
  return { worker, api: new InferenceClient(() => worker) };
};

const answer = (id: string) => ({
  id,
  answers: {
    EI: { type: "choice", probabilities: { E: 0.9, I: 0.1 }, confidence: 0.9 },
  },
});

const settled = <T>(promise: Promise<T>): Promise<{ ok: T } | { error: Error }> =>
  promise.then(
    (value) => ({ ok: value }),
    (error: Error) => ({ error }),
  );

const pending = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

describe("InferenceClient.load", () => {
  it("reports the backend the worker actually built on", async () => {
    const { worker, api } = client();
    const loaded = api.load("https://example.test/models/laya/", "https://x/ort/", () => {});
    expect(worker.sent).toEqual([
      { type: "load", modelUrl: "https://example.test/models/laya/", wasmPaths: "https://x/ort/" },
    ]);
    worker.deliver({ type: "ready", backend: "webgpu" });
    expect(await loaded).toEqual({ backend: "webgpu", reason: undefined });

    const second = client();
    const loadedOnCpu = second.api.load("u", "w", () => {});
    second.worker.deliver({ type: "ready", backend: "wasm", reason: "fp16 に対応していません" });
    expect(await loadedOnCpu).toEqual({
      backend: "wasm",
      reason: "fp16 に対応していません",
    });
  });

  it("passes cached model reads through as progress", async () => {
    const { worker, api } = client();
    const seen: unknown[] = [];
    const loaded = api.load("u", "w", (event) => seen.push(event));
    worker.deliver({ type: "loadProgress", file: "encoder.onnx", loaded: 1, total: 1, cached: true });
    worker.deliver({ type: "ready", backend: "wasm" });
    await loaded;
    expect(seen).toEqual([
      { type: "loadProgress", file: "encoder.onnx", loaded: 1, total: 1, cached: true },
    ]);
  });

  it("rejects when the model cannot be loaded at all", async () => {
    const { worker, api } = client();
    const loaded = settled(api.load("u", "w", () => {}));
    worker.deliver({ type: "error", message: "モデルの取得に失敗しました" });
    expect(await loaded).toEqual({ error: new Error("モデルの取得に失敗しました") });
  });
});

describe("InferenceClient.analyze", () => {
  it("collects items and resolves them on done", async () => {
    const { worker, api } = client();
    const items: string[] = [];
    const result = api.analyze([{ id: "a", text: "a" }], {
      onProgress: () => {},
      onItem: (item) => items.push(item.id),
    });
    await pending();
    worker.deliver({ type: "item", runId: worker.runId, item: answer("a") });
    worker.deliver({
      type: "progress",
      runId: worker.runId,
      done: 1,
      total: 1,
      elapsedMs: 10,
      etaMs: 0,
      batch: 8,
      backend: "webgpu",
    });
    worker.deliver({ type: "done", runId: worker.runId, total: 1, backend: "webgpu" });
    expect((await result).map((item) => item.id)).toEqual(["a"]);
    expect(items).toEqual(["a"]);
  });

  it("drops the WebGPU items when a run restarts on the CPU", async () => {
    const { worker, api } = client();
    const restarts: RestartInfo[] = [];
    const seen: string[] = [];
    const result = api.analyze([{ id: "a", text: "a" }, { id: "b", text: "b" }], {
      onProgress: () => {},
      onItem: (item) => seen.push(item.id),
      onRestart: (info) => restarts.push(info),
    });
    await pending();
    worker.deliver({ type: "item", runId: worker.runId, item: answer("a") });

    worker.deliver({
      type: "restart",
      runId: worker.runId,
      backend: "wasm",
      reason: "WebGPU で推論できなかったため、CPU（WASM）で最初から推論します。",
      total: 2,
    });
    worker.deliver({ type: "item", runId: worker.runId, item: answer("a") });
    worker.deliver({ type: "item", runId: worker.runId, item: answer("b") });
    worker.deliver({ type: "done", runId: worker.runId, total: 2, backend: "wasm" });

    const items = await result;
    // The CPU run answered both tweets; the earlier WebGPU item is not counted
    // twice, and the provisional view was rewound rather than left on tweet "a".
    expect(items.map((item) => item.id)).toEqual(["a", "b"]);
    expect(seen).toEqual(["a", "a", "b"]);
    expect(restarts).toEqual([
      {
        backend: "wasm",
        reason: "WebGPU で推論できなかったため、CPU（WASM）で最初から推論します。",
        total: 2,
      },
    ]);
  });

  it("rejects the run when the worker reports an error", async () => {
    const { worker, api } = client();
    const result = settled(api.analyze([{ id: "a", text: "a" }], {
      onProgress: () => {},
      onItem: () => {},
    }));
    await pending();
    worker.deliver({ type: "error", runId: worker.runId, message: "CPU でも開始できませんでした" });
    expect(await result).toEqual({ error: new Error("CPU でも開始できませんでした") });
  });

  it("rejects a cancelled run and ignores items from other runs", async () => {
    const { worker, api } = client();
    const result = settled(api.analyze([{ id: "a", text: "a" }], {
      onProgress: () => {},
      onItem: () => {},
    }));
    await pending();
    worker.deliver({ type: "item", runId: worker.runId + 99, item: answer("zz") });
    worker.deliver({ type: "cancelled", runId: worker.runId });
    const outcome = await result;
    expect(outcome).toHaveProperty("error");
    if (outcome && "error" in outcome) {
      expect(outcome.error.message).toContain("キャンセル");
    }
  });

  it("rejects every pending run when the worker dies", async () => {
    const { worker, api } = client();
    const result = settled(api.analyze([{ id: "a", text: "a" }], {
      onProgress: () => {},
      onItem: () => {},
    }));
    await pending();
    worker.onerror?.({ message: "worker crashed" } as ErrorEvent);
    const outcome = await result;
    expect(outcome).toHaveProperty("error");
    if (outcome && "error" in outcome) {
      expect(outcome.error.message).toContain("worker crashed");
    }
  });
});
