import { AXES } from "./mbti.js";
import type {
  AxisAnswer,
  AxisId,
  AxisMeta,
  AxisResult,
  MbtiResult,
  RankedAxisFit,
  RankedType,
} from "./types.js";

export interface AnalyzedTweet {
  id: string;
  answers: Record<string, AxisAnswer>;
}

export interface AggregateOptions {
  /**
   * Minimum `|P(positive) - P(negative)|` for a tweet to count as an effective
   * vote. Lower-margin tweets still soften the average toward 0.5.
   */
  marginThreshold: number;
}

export const DEFAULT_AGGREGATE_OPTIONS: AggregateOptions = {
  marginThreshold: 0.1,
};

function round(value: number, digits = 4): number {
  if (!Number.isFinite(value)) return 0;
  const f = 10 ** digits;
  return Math.round(value * f) / f;
}

/** Keeps `digits` significant digits so tiny agreement products stay non-zero. */
function roundSig(value: number, digits = 6): number {
  if (value === 0) return 0;
  return Number(value.toPrecision(digits));
}

/**
 * Reads the two pole probabilities of one axis answer.
 *
 * Returns `null` when the model did not answer this axis with usable evidence: no
 * answer, not a two-way choice, or a pole probability that is missing or not a
 * finite number. Such an answer is excluded from the evidence rather than read as
 * a 0/0 draw, which is what turned a model failure into a confident 0.5.
 */
function readMargins(answer: AxisAnswer | undefined, axis: AxisMeta): { positive: number; negative: number } | null {
  if (answer === undefined || answer.type !== "choice") return null;
  const probabilities = answer.probabilities ?? {};
  const positive = probabilities[axis.positive.code];
  const negative = probabilities[axis.negative.code];
  if (!Number.isFinite(positive) || !Number.isFinite(negative)) return null;
  return { positive: positive as number, negative: negative as number };
}

/**
 * Aggregates per-tweet Laya answers into a four-letter type.
 *
 * Each axis is one two-way `choice` question. For every tweet we take the
 * signed margin `P(positive) - P(negative)` and accumulate the positive and
 * negative magnitudes separately. Decisive tweets therefore move the axis;
 * ambiguous tweets contribute little. The winning pole is the one with the
 * larger accumulated magnitude.
 *
 * Two different situations are reported separately instead of both collapsing
 * into a 0.5: answers the model failed to produce (`invalidAnswers`) and valid
 * answers that simply do not clear the margin threshold (`insufficientEvidence`).
 */
export function aggregate(
  analyzed: AnalyzedTweet[],
  options: AggregateOptions = DEFAULT_AGGREGATE_OPTIONS,
): MbtiResult {
  const accumulators = AXES.map((axis) => ({
    axis,
    positive: 0,
    negative: 0,
    votes: 0,
    neutralVotes: 0,
    confidenceSum: 0,
    counted: 0,
    invalidAnswers: 0,
    marginBuckets: [0, 0, 0, 0, 0],
    signedBuckets: [0, 0, 0, 0, 0],
  }));

  let usedTweets = 0;
  let invalidAnswers = 0;

  for (const item of analyzed) {
    let voted = false;
    for (const acc of accumulators) {
      const answer = item.answers[acc.axis.id];
      const margins = readMargins(answer, acc.axis);
      if (margins === null) {
        acc.invalidAnswers++;
        invalidAnswers++;
        continue;
      }
      const { positive, negative } = margins;
      const margin = positive - negative;
      const bucket = Math.min(4, Math.floor(Math.abs(margin) * 10));
      acc.marginBuckets[bucket]++;
      // Same evidence, but kept signed and spread over the whole -1..1 range so
      // the card can show how the posts spread along the axis itself, not only
      // how large their margins were.
      acc.signedBuckets[Math.min(4, Math.floor(((margin + 1) / 2) * 5))]++;
      if (Math.abs(margin) >= options.marginThreshold) {
        if (margin > 0) acc.positive += margin;
        else acc.negative += -margin;
        acc.votes++;
        voted = true;
      } else {
        acc.neutralVotes++;
      }
      // Probabilities are the evidence and were just checked; a bad confidence
      // only affects this display average, so it is neutralized to 0.5.
      const confidence = Number.isFinite(answer.confidence)
        ? Math.min(1, Math.max(0, answer.confidence))
        : 0.5;
      acc.confidenceSum += confidence;
      acc.counted++;
    }
    if (voted) usedTweets++;
  }

  const axes: AxisResult[] = accumulators.map((acc) => {
    const total = acc.positive + acc.negative;
    const positiveProbability = total > 0 ? acc.positive / total : 0.5;
    const chosen =
      positiveProbability >= 0.5 ? acc.axis.positive : acc.axis.negative;
    return {
      id: acc.axis.id,
      name: acc.axis.name,
      positive: acc.axis.positive,
      negative: acc.axis.negative,
      chosen,
      probability: round(Math.max(positiveProbability, 1 - positiveProbability)),
      positiveProbability: round(positiveProbability),
      votes: acc.votes,
      neutralVotes: acc.neutralVotes,
      avgConfidence:
        acc.counted > 0 ? round(acc.confidenceSum / acc.counted) : 0,
      marginBuckets: [...acc.marginBuckets],
      signedBuckets: [...acc.signedBuckets],
      invalidAnswers: acc.invalidAnswers,
    };
  });

  return {
    type: axes.map((axis) => axis.chosen.code).join(""),
    axes,
    analyzed: analyzed.length,
    usedTweets,
    ranking: rankTypes(axes),
    invalidAnswers,
    insufficientEvidence: usedTweets === 0,
  };
}

interface Candidate {
  type: string;
  negative: boolean[];
  match: number[];
  score: number;
  tieBreak: number;
}

function enumerateCandidates(axes: AxisResult[]): Candidate[] {
  const candidates: Candidate[] = [];
  for (let mask = 0; mask < 1 << axes.length; mask++) {
    const negative: boolean[] = [];
    const match: number[] = [];
    let score = 1;
    let tieBreak = 0;
    axes.forEach((axis, index) => {
      const weight = 2 ** (axes.length - 1 - index);
      const isNegative = (mask & weight) !== 0;
      const fit = isNegative ? 1 - axis.positiveProbability : axis.positiveProbability;
      negative.push(isNegative);
      match.push(fit);
      score *= fit;
      if (isNegative) tieBreak += weight;
    });
    const type = axes
      .map((axis, index) => (negative[index] ? axis.negative.code : axis.positive.code))
      .join("");
    candidates.push({ type, negative, match, score, tieBreak });
  }
  return candidates;
}

/**
 * Ranks all sixteen types from the four per-axis positive probabilities.
 *
 * Each axis is one independent Bernoulli observation, so the fit of a
 * candidate type is the naive-Bayes agreement product
 * `∏ P(letter | evidence)`. A weak axis contributes a small factor for either
 * of its letters, so it can only shuffle probability mass between the two
 * types that share the other three letters. Summing the sixteen products gives
 * `∏ (p + (1 - p)) = 1`, one factor pair per axis, so the products are already
 * a distribution and normalizing only clears floating-point drift; the largest
 * one wins. The argmax of the product picks the more likely pole on each axis,
 * so the top entry reproduces the same type (and the same positive-pole
 * tie-break) as `aggregate`. Exact ties fall back to the positive pole of the
 * earliest differing axis, which keeps the order deterministic.
 */
export function rankTypes(axes: AxisResult[]): RankedType[] {
  if (axes.length === 0) return [];

  const candidates = enumerateCandidates(axes);
  const total = candidates.reduce((sum, candidate) => sum + candidate.score, 0);
  const ordered = [...candidates].sort(
    (a, b) => b.score - a.score || a.tieBreak - b.tieBreak,
  );
  const top = ordered[0].negative;

  return ordered.map((candidate, index) => {
    const differsFromTop: AxisId[] = [];
    const fits: RankedAxisFit[] = axes.map((axis, axisIndex) => {
      if (candidate.negative[axisIndex] !== top[axisIndex]) {
        differsFromTop.push(axis.id);
      }
      return {
        id: axis.id,
        name: axis.name,
        pole: candidate.negative[axisIndex] ? axis.negative : axis.positive,
        matchProbability: round(candidate.match[axisIndex]),
      };
    });
    return {
      type: candidate.type,
      rank: index + 1,
      probability: round(candidate.score / total),
      score: roundSig(candidate.score),
      differsFromTop,
      flips: differsFromTop.length,
      axes: fits,
    };
  });
}
