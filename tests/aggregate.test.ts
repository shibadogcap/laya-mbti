import { describe, expect, it } from "vitest";
import type { AxisAnswer, AxisResult } from "../src/lib/types.js";
import { aggregate, rankTypes } from "../src/lib/aggregate.js";
import type { AnalyzedTweet } from "../src/lib/aggregate.js";
import { AXES } from "../src/lib/mbti.js";

const choice = (
  pick: string,
  probabilities: Record<string, number>,
  confidence: number,
): AxisAnswer =>
  ({
    type: "choice",
    choice: pick,
    probabilities,
    confidence,
    action: { act_probability: 0.5 },
  }) as AxisAnswer;

function tweet(
  id: string,
  answers: Record<string, AxisAnswer>,
): AnalyzedTweet {
  return { id, answers };
}

/** Builds axis results with the given positive-pole probabilities, in AXES order. */
function axisResults(positiveProbabilities: number[]): AxisResult[] {
  return AXES.map((axis, index) => {
    const positiveProbability = positiveProbabilities[index];
    const chosen = positiveProbability >= 0.5 ? axis.positive : axis.negative;
    return {
      id: axis.id,
      name: axis.name,
      positive: axis.positive,
      negative: axis.negative,
      chosen,
      probability: Math.max(positiveProbability, 1 - positiveProbability),
      positiveProbability,
      votes: 1,
      neutralVotes: 0,
      avgConfidence: 0.5,
      invalidAnswers: 0,
    };
  });
}

describe("aggregate", () => {
  it("keeps a signed distribution along the axis, not just the margin size", () => {
    // Three posts lean E by different amounts and one leans I, so the signed
    // buckets have to separate the side as well as the strength.
    const result = aggregate([
      tweet("1", {
        EI: choice("E", { E: 0.9, I: 0.1 }, 0.9),
        SN: choice("S", { S: 0.8, N: 0.2 }, 0.8),
        TF: choice("T", { T: 0.7, F: 0.3 }, 0.7),
        JP: choice("J", { J: 0.7, P: 0.3 }, 0.7),
      }),
      tweet("2", {
        EI: choice("E", { E: 0.62, I: 0.38 }, 0.6),
        SN: choice("S", { S: 0.8, N: 0.2 }, 0.8),
        TF: choice("T", { T: 0.7, F: 0.3 }, 0.7),
        JP: choice("J", { J: 0.7, P: 0.3 }, 0.7),
      }),
      tweet("3", {
        EI: choice("E", { E: 0.55, I: 0.45 }, 0.5),
        SN: choice("S", { S: 0.8, N: 0.2 }, 0.8),
        TF: choice("T", { T: 0.7, F: 0.3 }, 0.7),
        JP: choice("J", { J: 0.7, P: 0.3 }, 0.7),
      }),
      tweet("4", {
        EI: choice("I", { E: 0.2, I: 0.8 }, 0.8),
        SN: choice("S", { S: 0.8, N: 0.2 }, 0.8),
        TF: choice("T", { T: 0.7, F: 0.3 }, 0.7),
        JP: choice("J", { J: 0.7, P: 0.3 }, 0.7),
      }),
    ]);
    const ei = result.axes[0];
    const signed = ei.signedBuckets!;

    expect(signed).toHaveLength(5);
    expect(signed.reduce((sum, count) => sum + count, 0)).toBe(4);
    // Buckets run from the negative pole (index 0) to the positive one (4), so
    // the I-leaning post is the only one left of the middle.
    expect(signed.filter((_count, index) => index <= 1).reduce((a, b) => a + b, 0)).toBe(1);
    expect(signed.filter((_count, index) => index >= 2).reduce((a, b) => a + b, 0)).toBe(3);
  });

  it("picks the pole with the larger accumulated margin", () => {
    const result = aggregate([
      tweet("1", {
        EI: choice("E", { E: 0.9, I: 0.1 }, 0.9),
        SN: choice("S", { S: 0.8, N: 0.2 }, 0.8),
        TF: choice("T", { T: 0.7, F: 0.3 }, 0.7),
        JP: choice("J", { J: 0.7, P: 0.3 }, 0.7),
      }),
      tweet("2", {
        EI: choice("I", { E: 0.4, I: 0.6 }, 0.6),
        SN: choice("N", { S: 0.45, N: 0.55 }, 0.3),
        TF: choice("F", { T: 0.2, F: 0.8 }, 0.8),
        JP: choice("P", { J: 0.1, P: 0.9 }, 0.9),
      }),
    ]);
    expect(result.type).toBe("ESFP");
    expect(result.usedTweets).toBe(2);
  });

  it("counts low-margin tweets as retained, not effective", () => {
    const result = aggregate([
      tweet("1", { SN: choice("N", { S: 0.475, N: 0.525 }, 0.2) }),
      tweet("2", { SN: choice("S", { S: 0.8, N: 0.2 }, 0.8) }),
    ]);
    const sn = result.axes.find((axis) => axis.id === "SN");
    expect(sn?.votes).toBe(1);
    expect(sn?.neutralVotes).toBe(1);
  });

  it("records the distribution of per-tweet decision margins", () => {
    const result = aggregate([
      tweet("1", { SN: choice("N", { S: 0.05, N: 0.95 }, 0.9) }),
      tweet("2", { SN: choice("N", { S: 0.625, N: 0.375 }, 0.7) }),
      tweet("3", { SN: choice("N", { S: 0.54, N: 0.46 }, 0.4) }),
    ]);
    const sn = result.axes.find((axis) => axis.id === "SN");
    expect(sn?.marginBuckets).toEqual([1, 0, 1, 0, 1]);
  });

  it("keeps confidence finite when a provider omits it", () => {
    const result = aggregate([
      tweet("1", { EI: choice("E", { E: 0.9, I: 0.1 }, Number.NaN) }),
    ]);
    const ei = result.axes.find((axis) => axis.id === "EI");
    expect(Number.isFinite(ei?.avgConfidence)).toBe(true);
    expect(ei?.avgConfidence).toBe(0.5);
  });

  it("falls back to an even split with no effective votes", () => {
    const result = aggregate([
      tweet("1", { EI: choice("E", { E: 0.51, I: 0.49 }, 0.02) }),
    ]);
    const ei = result.axes.find((axis) => axis.id === "EI");
    expect(ei?.votes).toBe(0);
    expect(ei?.positiveProbability).toBe(0.5);
    expect(result.usedTweets).toBe(0);
  });
});

describe("aggregate failure reporting", () => {
  it("counts a missing answer as invalid instead of a 0/0 draw", () => {
    const result = aggregate([tweet("1", { EI: choice("E", { E: 0.9, I: 0.1 }, 0.9) })]);
    const ei = result.axes.find((axis) => axis.id === "EI");
    const sn = result.axes.find((axis) => axis.id === "SN");

    expect(ei?.invalidAnswers).toBe(0);
    expect(ei?.votes).toBe(1);
    expect(sn?.invalidAnswers).toBe(1);
    expect(sn?.votes).toBe(0);
    expect(result.invalidAnswers).toBe(3);
    expect(result.usedTweets).toBe(1);
    expect(result.insufficientEvidence).toBe(false);
  });

  it("rejects non-finite probabilities instead of scoring them as 0.5", () => {
    const result = aggregate([
      tweet("1", {
        SN: choice("N", { S: Number.NaN, N: 0.6 }, 0.9),
        TF: choice("F", { T: 0.4, F: 0.6 }, 0.9),
        JP: choice("P", { J: 0.1, P: 0.9 }, 0.9),
      }),
    ]);
    const sn = result.axes.find((axis) => axis.id === "SN");
    const tf = result.axes.find((axis) => axis.id === "TF");

    expect(sn?.invalidAnswers).toBe(1);
    expect(sn?.votes).toBe(0);
    expect(sn?.marginBuckets).toEqual([0, 0, 0, 0, 0]);
    expect(result.invalidAnswers).toBe(2);
    // The failed axis falls back to 0.5; the valid axes still decide the type.
    expect(sn?.positiveProbability).toBe(0.5);
    expect(tf?.votes).toBe(1);
    expect(tf?.positiveProbability).toBe(0);
    expect(result.type).toBe("ESFP");
  });

  it("keeps a failed axis out of the confidence average", () => {
    const result = aggregate([
      tweet("1", {
        SN: choice("N", { S: Number.NaN, N: 0.6 }, 0.9),
        TF: choice("F", { T: 0.4, F: 0.6 }, 0.9),
      }),
    ]);
    const sn = result.axes.find((axis) => axis.id === "SN");
    const tf = result.axes.find((axis) => axis.id === "TF");

    expect(sn?.avgConfidence).toBe(0);
    expect(tf?.avgConfidence).toBe(0.9);
  });

  it("separates zero evidence from invalid answers", () => {
    const validButNeutral = aggregate([
      tweet("1", {
        EI: choice("E", { E: 0.51, I: 0.49 }, 0.05),
        SN: choice("N", { S: 0.5, N: 0.5 }, 0),
        TF: choice("F", { T: 0.5, F: 0.5 }, 0),
        JP: choice("P", { J: 0.5, P: 0.5 }, 0),
      }),
    ]);

    expect(validButNeutral.invalidAnswers).toBe(0);
    expect(validButNeutral.insufficientEvidence).toBe(true);
    expect(validButNeutral.axes.every((axis) => axis.avgConfidence === 0)).toBe(false);

    const modelFailure = aggregate([
      tweet("1", {
        EI: choice("E", { E: Number.NaN, I: Number.NaN }, Number.NaN),
        SN: { type: "score", probabilities: {}, confidence: Number.NaN } as unknown as AxisAnswer,
        TF: { type: "choice", probabilities: { T: Number.NaN, F: 1 }, confidence: 1 } as unknown as AxisAnswer,
        JP: undefined as unknown as AxisAnswer,
      }),
    ]);

    expect(modelFailure.invalidAnswers).toBe(4);
    expect(modelFailure.insufficientEvidence).toBe(true);
    expect(modelFailure.usedTweets).toBe(0);
    expect(modelFailure.axes.every((axis) => axis.avgConfidence === 0)).toBe(true);
    expect(modelFailure.axes.every((axis) => axis.votes === 0)).toBe(true);
    // Nothing was measured, so the ranking is uniform and the type is the default.
    expect(modelFailure.ranking[0].probability).toBeCloseTo(1 / 16, 6);
  });
});

describe("rankTypes", () => {
  it("ranks the primary type first, then the runner-up and third place", () => {
    // EI .9 / SN .7 / TF .85 / JP .65 -> ESTJ leads, and the single flips of
    // the two weakest axes take the next two places.
    const ranking = rankTypes(axisResults([0.9, 0.7, 0.85, 0.65]));

    expect(ranking.slice(0, 3).map((entry) => entry.type)).toEqual([
      "ESTJ",
      "ESTP",
      "ENTJ",
    ]);
    expect(ranking[0].probability).toBeCloseTo(0.348075, 4);
    expect(ranking[1].probability).toBeCloseTo(0.187425, 4);
    expect(ranking[2].probability).toBeCloseTo(0.149175, 4);
  });

  it("scores each type as the product of its per-axis agreements", () => {
    const ranking = rankTypes(axisResults([0.9, 0.7, 0.85, 0.65]));
    const top = ranking[0];

    expect(top.score).toBeCloseTo(0.9 * 0.7 * 0.85 * 0.65, 6);
    expect(top.axes.map((fit) => fit.matchProbability)).toEqual([
      0.9, 0.7, 0.85, 0.65,
    ]);
    expect(top.axes.map((fit) => fit.pole.code).join("")).toBe(top.type);
  });

  it("spreads the sixteen probabilities over a normalized distribution", () => {
    const ranking = rankTypes(axisResults([0.9, 0.7, 0.85, 0.65]));

    expect(ranking).toHaveLength(16);
    expect(ranking.reduce((sum, entry) => sum + entry.probability, 0)).toBeCloseTo(
      1,
      3,
    );
    expect(ranking.map((entry) => entry.rank)).toEqual(
      Array.from({ length: 16 }, (_, index) => index + 1),
    );
    for (let i = 1; i < ranking.length; i++) {
      expect(ranking[i].probability).toBeLessThanOrEqual(ranking[i - 1].probability);
    }
  });

  it("reports which axes each alternative flips against the top type", () => {
    const ranking = rankTypes(axisResults([0.9, 0.7, 0.85, 0.65]));

    expect(ranking[0].flips).toBe(0);
    expect(ranking[0].differsFromTop).toEqual([]);
    expect(ranking[1].differsFromTop).toEqual(["JP"]);
    expect(ranking[1].flips).toBe(1);
    expect(ranking[2].differsFromTop).toEqual(["SN"]);
  });

  it("weights axes by strength instead of counting flips", () => {
    // ENTP flips two axes but still beats ESFJ, which flips only TF: the
    // 0.7 evidence for S outweighs the single weaker disagreement on TF.
    const ranking = rankTypes(axisResults([0.9, 0.7, 0.85, 0.65]));
    const byType = new Map(ranking.map((entry) => [entry.type, entry]));

    expect(byType.get("ENTP")!.flips).toBe(2);
    expect(byType.get("ESFJ")!.flips).toBe(1);
    expect(ranking.indexOf(byType.get("ENTP")!)).toBeLessThan(
      ranking.indexOf(byType.get("ESFJ")!),
    );
  });

  it("breaks exact ties toward the positive pole, matching aggregate", () => {
    const ranking = rankTypes(axisResults([0.5, 0.5, 0.5, 0.5]));

    for (const entry of ranking) {
      expect(entry.probability).toBeCloseTo(1 / 16, 6);
    }
    expect(ranking.map((entry) => entry.type)).toEqual([
      "ESTJ",
      "ESTP",
      "ESFJ",
      "ESFP",
      "ENTJ",
      "ENTP",
      "ENFJ",
      "ENFP",
      "ISTJ",
      "ISTP",
      "ISFJ",
      "ISFP",
      "INTJ",
      "INTP",
      "INFJ",
      "INFP",
    ]);
  });

  it("keeps tiny agreement products non-zero", () => {
    const ranking = rankTypes(axisResults([0.9999, 0.0001, 0.0001, 0.9999]));
    const bottom = ranking[ranking.length - 1];

    expect(bottom.type).toBe("ISTP");
    expect(bottom.score).toBeGreaterThan(0);
    expect(bottom.probability).toBe(0);
  });
});

describe("aggregate ranking", () => {
  it("puts the reported type first and a weaker type second", () => {
    const result = aggregate([
      tweet("1", {
        EI: choice("E", { E: 0.9, I: 0.1 }, 0.9),
        SN: choice("S", { S: 0.8, N: 0.2 }, 0.8),
        TF: choice("T", { T: 0.7, F: 0.3 }, 0.7),
        JP: choice("J", { J: 0.7, P: 0.3 }, 0.7),
      }),
      tweet("2", {
        EI: choice("I", { E: 0.4, I: 0.6 }, 0.6),
        SN: choice("N", { S: 0.45, N: 0.55 }, 0.3),
        TF: choice("F", { T: 0.2, F: 0.8 }, 0.8),
        JP: choice("P", { J: 0.1, P: 0.9 }, 0.9),
      }),
    ]);

    // SN has no dissent, TF ends at 0.4 T vs 0.6 F, JP at 0.33 J vs 0.67 P.
    expect(result.type).toBe("ESFP");
    expect(result.ranking[0].type).toBe("ESFP");
    expect(result.ranking[0].flips).toBe(0);
    expect(result.ranking[1].type).toBe("ESTP");
    expect(result.ranking[1].differsFromTop).toEqual(["TF"]);
    expect(result.ranking[1].probability).toBeGreaterThan(
      result.ranking[2].probability,
    );
  });

  it("ranks all sixteen types evenly with no evidence", () => {
    const result = aggregate([]);

    expect(result.type).toBe("ESTJ");
    expect(result.ranking).toHaveLength(16);
    expect(result.ranking[0].type).toBe(result.type);
    for (const entry of result.ranking) {
      expect(entry.probability).toBeCloseTo(1 / 16, 6);
    }
  });
});
