import type { AxisAnswer } from "./types.js";

/**
 * Names the first value in an answer that cannot be aggregated, or null when
 * every probability and the confidence are finite.
 *
 * Kept separate from the agent so the worker can run the same check on a
 * one-post probe before a real run starts: a phone GPU that overflows fp16
 * activations to `inf` produces NaN in the head, and finding that out after a
 * few hundred posts wastes the whole wait.
 */
export function isFiniteAnswer(answer: unknown): string | null {
  if (typeof answer !== "object" || answer === null) return "answer がオブジェクトではありません";
  const record = answer as Partial<AxisAnswer>;
  const probabilities = record.probabilities;
  if (typeof probabilities !== "object" || probabilities === null) {
    return "確率がありません";
  }
  for (const [key, value] of Object.entries(probabilities)) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      return `${key} = ${describe(value)}`;
    }
  }
  if (typeof record.confidence !== "number" || !Number.isFinite(record.confidence)) {
    return `confidence = ${describe(record.confidence)}`;
  }
  return null;
}

function describe(value: unknown): string {
  if (typeof value !== "number") return `${typeof value} (${String(value)})`;
  if (Number.isNaN(value)) return "NaN";
  return value > 0 ? "+Infinity" : "-Infinity";
}
