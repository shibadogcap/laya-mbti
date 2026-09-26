import { afterEach, describe, expect, it, vi } from "vitest";
import {
  assetIsCached,
  decideWasmFallback,
  isModelOutputError,
  nextBackend,
  wasmPlan,
  wasmThreadCount,
  webGpuPlan,
} from "../src/vendor/laya-ts/providers.js";

const deviceLost = () => new Error("Device lost");
const nanGuard = () =>
  new Error("モデルが質問「EI」（行 0）に NaN を返しました。答えを計算できないため、集計しません。");

describe("provider plans", () => {
  it("prefers WebGPU for the encoder, with WASM left for the small head", () => {
    const plan = webGpuPlan(1);
    expect(plan.backend).toBe("webgpu");
    expect(plan.encoderProviders).toEqual(["webgpu"]);
    expect(plan.headProviders).toEqual(["webgpu", "wasm"]);
    // The WebGPU kernel of the fused LayerNorm rejects this graph, so the
    // encoder session must stay unoptimized.
    expect(plan.encoderGraphOptimizationLevel).toBe("disabled");
    expect(plan.statesPerChunk).toBe(8);
  });

  it("puts both sessions on WASM for the fallback", () => {
    const plan = wasmPlan(4);
    expect(plan.backend).toBe("wasm");
    expect(plan.encoderProviders).toEqual(["wasm"]);
    expect(plan.headProviders).toEqual(["wasm"]);
    // No GPU is touched at all on this path, so nothing can fail over to WebGPU.
    expect([...plan.encoderProviders, ...plan.headProviders]).not.toContain("webgpu");
    expect(plan.numThreads).toBe(4);
  });

  it("runs the CPU encoder one tweet at a time", () => {
    // A wide batch buys GPU occupancy; on the CPU it only raises peak memory.
    expect(wasmPlan().statesPerChunk).toBe(1);
    expect(webGpuPlan().statesPerChunk).toBeGreaterThan(1);
  });

  it("keeps the disabled optimization level a WebGPU-only workaround", () => {
    // The CPU fusions are what make an fp16 encoder usable at all.
    expect(wasmPlan().encoderGraphOptimizationLevel).not.toBe("disabled");
    expect(wasmPlan().encoderGraphOptimizationLevel).toBe("basic");
  });
});

describe("wasmThreadCount", () => {
  it("stays single-threaded without SharedArrayBuffer", () => {
    // More than one WASM thread needs COOP/COEP; ORT throws instead of dropping
    // to one thread, so this is a requirement of the host page, not a tuning knob.
    expect(wasmThreadCount({ crossOriginIsolated: false, hardwareConcurrency: 16 })).toBe(1);
    expect(wasmThreadCount({ hardwareConcurrency: 16 })).toBe(1);
  });

  it("uses spare cores once the page is cross-origin isolated", () => {
    expect(wasmThreadCount({ crossOriginIsolated: true, hardwareConcurrency: 8 })).toBe(4);
    expect(wasmThreadCount({ crossOriginIsolated: true, hardwareConcurrency: 3 })).toBe(2);
    expect(wasmThreadCount({ crossOriginIsolated: true, hardwareConcurrency: 1 })).toBe(1);
    expect(wasmThreadCount({ crossOriginIsolated: true, hardwareConcurrency: 0 })).toBe(1);
  });

  it("ignores a nonsensical core count", () => {
    expect(wasmThreadCount({ crossOriginIsolated: true, hardwareConcurrency: Number.NaN })).toBe(1);
    expect(wasmThreadCount({ crossOriginIsolated: true })).toBe(1);
  });
});

describe("decideWasmFallback", () => {
  it("falls back when the fp16 preflight refuses the GPU", () => {
    const decision = decideWasmFallback({
      error: new Error("この GPU は fp16 推論に対応していません（WebGPU の shader-f16 機能がありません）。"),
      phase: "preflight",
    });
    expect(decision.fallback).toBe(true);
    expect(decision.reason).toContain("CPU（WASM）");
    expect(decision.reason).toContain("shader-f16");
  });

  it("falls back when a WebGPU session cannot be created", () => {
    const decision = decideWasmFallback({
      error: new Error("WebGPU で encoder を初期化できませんでした。 (compile failed)"),
      phase: "session",
    });
    expect(decision.fallback).toBe(true);
    expect(decision.reason).toContain("最初から");
  });

  it("falls back when a batch dies at runtime, once the batch is already minimal", () => {
    for (const error of [
      deviceLost(),
      new Error("Failed to allocate: out of memory"),
      new Error("Buffer size (268435456) exceeds the max buffer size"),
      new Error("WebGPU error: validation error at shader compilation"),
      nanGuard(),
    ]) {
      expect(decideWasmFallback({ error, phase: "runtime" }).fallback).toBe(true);
    }
  });

  it("keeps runtime failures the CPU would raise identically", () => {
    for (const error of [
      new Error("question \"EI\": definition must be a dict, got list"),
      new Error("解析をキャンセルしました"),
      new Error("no items to collate"),
    ]) {
      const decision = decideWasmFallback({ error, phase: "runtime" });
      expect(decision.fallback).toBe(false);
      expect(decision.reason).toBe("");
    }
  });

  it("does not retry on the CPU when the model file itself is missing", () => {
    const decision = decideWasmFallback({
      error: new Error("fetch failed for /models/laya/encoder.onnx: 404"),
      phase: "session",
    });
    expect(decision.fallback).toBe(false);
  });
});

describe("nextBackend", () => {
  const fallBack = { fallback: true, reason: "CPU でやり直します" };
  const keep = { fallback: false, reason: "" };

  it("moves from WebGPU to WASM exactly once", () => {
    expect(nextBackend("webgpu", fallBack)).toBe("wasm");
    // WASM is the last backend: a second failure is the user's, not ours.
    expect(nextBackend("wasm", fallBack)).toBeNull();
  });

  it("stays put when the failure is not a backend failure", () => {
    expect(nextBackend("webgpu", keep)).toBeNull();
  });
});

describe("isModelOutputError", () => {
  it("recognises the guards the Agent raises", () => {
    expect(isModelOutputError(nanGuard())).toBe(true);
    expect(isModelOutputError(new Error("モデルの出力行が不足しています"))).toBe(true);
    expect(isModelOutputError(new Error("モデルが質問「EI」の確率の合計が 0 です"))).toBe(true);
  });

  it("leaves backend and user failures alone", () => {
    expect(isModelOutputError(deviceLost())).toBe(false);
    expect(isModelOutputError(new Error("解析をキャンセルしました"))).toBe(false);
  });
});

describe("assetIsCached", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("reports a Cache Storage hit, so a rebuild downloads nothing", async () => {
    const match = vi.fn(async (url: string) => (url.endsWith("encoder.onnx") ? { id: 1 } : undefined));
    vi.stubGlobal("caches", { match });
    expect(await assetIsCached("https://example.test/models/laya/encoder.onnx")).toBe(true);
    expect(await assetIsCached("https://example.test/models/laya/head.onnx")).toBe(false);
  });

  it("reports a miss when Cache Storage is unavailable or blocked", async () => {
    vi.stubGlobal("caches", undefined);
    expect(await assetIsCached("https://example.test/models/laya/encoder.onnx")).toBe(false);

    vi.stubGlobal("caches", {
      match: async () => {
        throw new Error("quota exceeded");
      },
    });
    expect(await assetIsCached("https://example.test/models/laya/encoder.onnx")).toBe(false);
  });
});
