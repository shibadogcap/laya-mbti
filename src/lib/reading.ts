import { AXES } from "./mbti.js";
import { decodeEntities } from "./parse.js";
import type { AnalyzedItem } from "./inference.js";

type AxisMeta = (typeof AXES)[number];

export interface PoleReading {
  label: string;
  code: string;
  share: number;
}

export const TWEET_TIME = new Intl.DateTimeFormat("ja-JP", {
  year: "numeric",
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
});

export const TWEET_COUNT = new Intl.NumberFormat("ja-JP");

/**
 * One-line preview of a post. Newlines are kept on purpose: how someone breaks
 * their lines is part of how they write, and collapsing it into a single line
 * turned a two-line post into an unreadable strip. Runs of spaces and tabs are
 * still squeezed so padding does not leak into the card.
 *
 * Entities are decoded again here on purpose. The parser already does it, so this
 * is a no-op for archive text, but it keeps the displayed string free of `&amp;`
 * even if a text arrives from a path that skipped the parser.
 */
export function shortText(text: string): string {
  const normalized = decodeEntities(text)
    .replace(/\r\n?/g, "\n")
    .replace(/[ \t\u3000]+/g, " ")
    .replace(/ *\n */g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  return normalized.length > 200 ? `${normalized.slice(0, 200)}…` : normalized;
}

/** The pole the model picked on one axis, with the confidence it picked it with. */
export function chosenPole(axis: AxisMeta, item: AnalyzedItem): PoleReading | null {
  const answer = item.answers[axis.id];
  if (!answer) return null;
  const positive = answer.probabilities[axis.positive.code] ?? 0;
  const negative = answer.probabilities[axis.negative.code] ?? 0;
  const pole = positive >= negative ? axis.positive : axis.negative;
  return {
    label: `#${pole.label.replace(/\s*\([^)]*\)\s*$/, "")}`,
    code: pole.code,
    share: Math.round(Math.min(100, Math.max(0, Math.max(positive, negative) * 100))),
  };
}

/** The four letters the model read out of one post, e.g. `ENFP`. */
export function itemType(item: AnalyzedItem): string {
  return AXES.map((axis) => {
    const probabilities = item.answers[axis.id]?.probabilities ?? {};
    return (probabilities[axis.positive.code] ?? 0) >= (probabilities[axis.negative.code] ?? 0)
      ? axis.positive.code
      : axis.negative.code;
  }).join("");
}

/** Every axis reading of one post, in the fixed axis order. */
export function itemPoles(item: AnalyzedItem): { axis: AxisMeta; pole: PoleReading }[] {
  return AXES.map((axis) => ({ axis, pole: chosenPole(axis, item) })).filter(
    (entry): entry is { axis: AxisMeta; pole: PoleReading } => entry.pole !== null,
  );
}
