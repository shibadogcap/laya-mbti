import { describe, expect, it } from "vitest";
import { isFiniteAnswer } from "../src/lib/finite.js";

const answer = (probabilities: Record<string, number>, confidence = 0.8) => ({
  type: "choice",
  choice: "E",
  probabilities,
  confidence,
});

describe("isFiniteAnswer", () => {
  it("passes a normal answer", () => {
    expect(isFiniteAnswer(answer({ E: 0.7, I: 0.3 }))).toBeNull();
  });

  it("names the pole that overflowed", () => {
    expect(isFiniteAnswer(answer({ E: Number.NaN, I: 0.3 }))).toBe("E = NaN");
    expect(isFiniteAnswer(answer({ E: Number.POSITIVE_INFINITY, I: 0.3 }))).toBe(
      "E = +Infinity",
    );
    expect(isFiniteAnswer(answer({ E: Number.NEGATIVE_INFINITY, I: 0.3 }))).toBe(
      "E = -Infinity",
    );
  });

  it("checks the confidence too", () => {
    expect(isFiniteAnswer(answer({ E: 0.5, I: 0.5 }, Number.NaN))).toBe("confidence = NaN");
  });

  it("reports a missing or malformed probability map", () => {
    expect(isFiniteAnswer({ type: "choice", choice: "E", confidence: 0.5 })).toBe(
      "確率がありません",
    );
    expect(isFiniteAnswer(null)).toBe("answer がオブジェクトではありません");
    expect(isFiniteAnswer("E")).toBe("answer がオブジェクトではありません");
  });
});
