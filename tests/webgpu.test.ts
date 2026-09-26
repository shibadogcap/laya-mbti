import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ENCODER_REQUIRED_FEATURES,
  checkEncoderWebGpu,
  encoderCapability,
  errorText,
  isBatchSizeError,
  isMemoryError,
} from "../src/vendor/laya-ts/providers.js";

const supports = (...features: string[]) => ({ has: (f: string) => features.includes(f) });

describe("encoderCapability", () => {
  it("passes when the adapter exposes the fp16 feature", () => {
    const report = encoderCapability({ gpu: {}, features: supports("shader-f16") });
    expect(report.ok).toBe(true);
    expect(report.reason).toBe("ok");
  });

  it("explains a missing shader-f16 in Japanese and names the feature", () => {
    const report = encoderCapability({ gpu: {}, features: supports() });
    expect(report.ok).toBe(false);
    expect(report.reason).toBe("missing-feature");
    expect(report.feature).toBe("shader-f16");
    expect(report.message).toContain("shader-f16");
    expect(report.message).toContain("fp16");
  });

  it("explains a missing WebGPU implementation", () => {
    const report = encoderCapability({ gpu: null });
    expect(report.reason).toBe("no-webgpu");
    expect(report.message).toContain("WebGPU");
  });

  it("explains a missing adapter", () => {
    const report = encoderCapability({ gpu: {}, features: null });
    expect(report.reason).toBe("no-adapter");
    expect(report.message).toContain("アダプター");
  });

  it("requires exactly the fp16 encoder feature", () => {
    expect(ENCODER_REQUIRED_FEATURES).toContain("shader-f16");
  });
});

describe("checkEncoderWebGpu", () => {
  afterEach(() => vi.unstubAllGlobals());

  const withGpu = (gpu: unknown) => vi.stubGlobal("navigator", { gpu });

  it("passes for an adapter with shader-f16", async () => {
    withGpu({ requestAdapter: async () => ({ features: supports("shader-f16") }) });
    expect((await checkEncoderWebGpu()).ok).toBe(true);
  });

  it("reports a missing adapter when requestAdapter throws", async () => {
    withGpu({
      requestAdapter: async () => {
        throw new Error("adapter request denied");
      },
    });
    const report = await checkEncoderWebGpu();
    expect(report.reason).toBe("no-adapter");
    expect(report.message).toContain("アダプター");
  });

  it("reports a missing feature from the adapter", async () => {
    withGpu({ requestAdapter: async () => ({ features: supports() }) });
    expect((await checkEncoderWebGpu()).reason).toBe("missing-feature");
  });

  it("reports a browser without WebGPU", async () => {
    withGpu(undefined);
    expect((await checkEncoderWebGpu()).reason).toBe("no-webgpu");
  });
});

describe("errorText", () => {
  it("keeps a distinguishing error name", () => {
    const error = new Error("Device lost");
    error.name = "GPUDeviceLostError";
    expect(errorText(error)).toBe("GPUDeviceLostError: Device lost");
  });

  it("accepts a plain string or a message-less error", () => {
    expect(errorText("boom")).toBe("boom");
    expect(errorText(new Error())).toBe("Error");
  });
});

describe("batch-size error classification", () => {
  it("keeps recognizing out-of-memory failures", () => {
    expect(isMemoryError(new Error("Failed to allocate: out of memory"))).toBe(true);
    expect(isMemoryError(new Error("OOM while copying tensor"))).toBe(true);
    expect(isMemoryError(new Error("device lost"))).toBe(false);
  });

  it("catches WebGPU limit, validation and device-lost messages", () => {
    const retryable = [
      "Device lost",
      "GPUDeviceLostError: device was lost",
      "Buffer size (268435456) exceeds the max buffer size",
      "Maximum number of bindings exceeded",
      "WebGPU error: validation error at shader compilation",
      "dispatchWorkgroups out of bounds",
      "Buffer is too large for the device",
      "Failed to request an adapter",
      "insufficient resources for the allocation",
    ];
    for (const message of retryable) {
      expect(isBatchSizeError(new Error(message))).toBe(true);
    }
  });

  it("leaves unrelated failures to the caller", () => {
    expect(isBatchSizeError(new Error("fetch failed for /models/laya/encoder.onnx: 404"))).toBe(
      false,
    );
    expect(isBatchSizeError(new Error("解析をキャンセルしました"))).toBe(false);
  });

  it("reads a non-Error value as text", () => {
    expect(isBatchSizeError("out of memory")).toBe(true);
  });
});
