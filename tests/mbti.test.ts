import { describe, expect, it } from "vitest";
import { AXES, buildQuestions, randomizedQuestionPlan, shuffle } from "../src/lib/mbti.js";

const ids = () => AXES.map((axis) => axis.id);

describe("buildQuestions", () => {
  it("keeps the canonical axis order by default", () => {
    expect(Object.keys(buildQuestions())).toEqual(ids());
  });

  it("emits the axes in the requested order", () => {
    const order = ["JP", "EI", "TF", "SN"] as const;
    expect(Object.keys(buildQuestions(order))).toEqual([...order]);
  });

  it("lists the positive pole first unless the axis is flipped", () => {
    const flipped = new Set<"EI">(["EI"]);
    const questions = buildQuestions(ids(), flipped);
    expect(Object.keys(questions.EI.criteria as object)).toEqual(["I", "E"]);
    expect(Object.keys(questions.SN.criteria as object)).toEqual(["S", "N"]);
  });

  it("skips an unknown axis id instead of emitting a broken question", () => {
    const questions = buildQuestions(["EI", "ZZ"] as never);
    expect(Object.keys(questions)).toEqual(["EI"]);
  });
});

describe("shuffle", () => {
  it("returns a permutation and leaves the input alone", () => {
    const input = [1, 2, 3, 4, 5];
    const out = shuffle(input, () => 0.42);
    expect([...out].sort()).toEqual(input);
    expect(input).toEqual([1, 2, 3, 4, 5]);
    expect(out).not.toBe(input);
  });

  it("is deterministic for a given random source", () => {
    const fixed = () => 0.5;
    expect(shuffle([1, 2, 3, 4], fixed)).toEqual(shuffle([1, 2, 3, 4], fixed));
  });
});

describe("randomizedQuestionPlan", () => {
  it("keeps every axis exactly once", () => {
    for (let seed = 0; seed < 50; seed++) {
      const plan = randomizedQuestionPlan(() => (seed * 0.0197) % 1);
      expect([...plan.order].sort()).toEqual([...ids()].sort());
      expect(plan.order).toHaveLength(4);
    }
  });

  it("flips some axes and leaves the rest alone", () => {
    let sawFlip = false;
    let sawNoFlip = false;
    for (let seed = 0; seed < 50 && !(sawFlip && sawNoFlip); seed++) {
      const plan = randomizedQuestionPlan(() => (seed * 0.0197) % 1);
      if (plan.flipOptions.size > 0) sawFlip = true;
      if (plan.flipOptions.size < 4) sawNoFlip = true;
    }
    expect(sawFlip).toBe(true);
    expect(sawNoFlip).toBe(true);
  });

  it("changes the axis order across seeds", () => {
    const orders = new Set<string>();
    for (let seed = 0; seed < 40; seed++) {
      orders.add(randomizedQuestionPlan(() => (seed * 0.0197) % 1).order.join(""));
    }
    expect(orders.size).toBeGreaterThan(1);
  });
});
