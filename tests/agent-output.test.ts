import { describe, expect, it } from "vitest";
import {
  Agent,
  validateAnswer,
  validateHeadOutput,
  type ChoiceAnswer,
  type NoulAnswer,
  type SystemAnswer,
} from "../src/vendor/laya-ts/agent.js";
import type { Batch, EncodedHidden, SessionProvider } from "../src/vendor/laya-ts/providers.js";
import { buildQuestions } from "../src/lib/mbti.js";
import type { QuestionDef } from "../src/vendor/laya-ts/agent.js";

const questions = buildQuestions();
const ids = Object.keys(questions);
const options = 2;

const hidden: EncodedHidden = { data: new Float32Array(0), dims: [0, 1, 0] };

/** Provider stub: the encoder is a no-op, the head returns whatever rows the test wants. */
function stubProvider(head: () => { logits: number[][]; act: number[][] }): SessionProvider {
  return {
    runEncoder: async (): Promise<{ lastHidden: EncodedHidden }> => ({ lastHidden: hidden }),
    runHead: async (_h: EncodedHidden, _b: Batch) => head(),
  };
}

/** One all-valid row per (state, question), in collate order. */
function healthyRows(states: number): { logits: number[][]; act: number[][] } {
  const logits: number[][] = [];
  for (let s = 0; s < states; s++) {
    for (let j = 0; j < ids.length; j++) {
      logits.push(j % 2 === 0 ? [2, -2] : [-2, 2]);
    }
  }
  return { logits, act: logits.map((row) => [...row]) };
}

const batchAgent = (head: () => { logits: number[][]; act: number[][] }) =>
  new Agent({ provider: stubProvider(head) });

const choiceAnswer = (over: Partial<ChoiceAnswer> = {}): SystemAnswer =>
  ({
    type: "choice",
    choice: "E",
    probabilities: { E: 0.9, I: 0.1 },
    confidence: 0.9,
    answer_confidence: 0.9,
    action: { act_probability: 0.5 },
    ...over,
  }) as SystemAnswer;

describe("validateHeadOutput", () => {
  const rows = ids.map((id) => ({ id, options }));

  it("accepts a full, finite output", () => {
    expect(validateHeadOutput([[1, 2], [3, 4]], [[0, 1], [1, 0]], rows.slice(0, 2))).toBeNull();
  });

  it("rejects non-array output", () => {
    const message = validateHeadOutput(undefined, [[0, 1]], rows.slice(0, 1));
    expect(message).toContain("壊れています");
  });

  it("rejects a truncated pass and names the expected row count", () => {
    const message = validateHeadOutput([[1, 2]], [[1, 0]], rows);
    expect(message).toContain("出力行が不足しています");
    expect(message).toContain(String(rows.length));
  });

  it("rejects a row shorter than the option count", () => {
    const message = validateHeadOutput([[1]], [[1, 0]], rows.slice(0, 1));
    expect(message).toContain("1 個");
  });

  it("rejects NaN and Infinity logits, naming the question", () => {
    expect(validateHeadOutput([[Number.NaN, 2]], [[1, 0]], rows.slice(0, 1))).toContain("NaN");
    expect(validateHeadOutput([[1, 2]], [[Number.POSITIVE_INFINITY, 0]], rows.slice(0, 1))).toContain(
      "EI",
    );
  });

  it("rejects an act row that cannot form a probability", () => {
    expect(validateHeadOutput([[1, 2]], [[1]], rows.slice(0, 1))).toContain("act");
    expect(validateHeadOutput([[1, 2]], [[Number.NaN, 0]], rows.slice(0, 1))).toContain("NaN");
  });

  it("ignores padded marker positions past the option count", () => {
    // The exported head returns one logit per batch-wide marker slot; only the
    // first `options` entries become an answer.
    expect(validateHeadOutput([[1, 2, -1e4, -1e4]], [[1, 0]], rows.slice(0, 1))).toBeNull();
  });
});

describe("validateAnswer", () => {
  it("accepts a well-formed answer", () => {
    expect(validateAnswer("EI", choiceAnswer(), options)).toBeNull();
  });

  it("rejects a missing answer", () => {
    expect(validateAnswer("EI", undefined, options)).toContain("返しませんでした");
  });

  it("rejects a wrong number of probabilities", () => {
    expect(validateAnswer("EI", choiceAnswer({ probabilities: { E: 1 } }), options)).toContain(
      "2 個",
    );
  });

  it("rejects non-finite probabilities and confidence", () => {
    expect(
      validateAnswer("EI", choiceAnswer({ probabilities: { E: Number.NaN, I: 0.1 } }), options),
    ).toContain("NaN");
    expect(validateAnswer("EI", choiceAnswer({ confidence: Number.NaN }), options)).toContain(
      "確信度",
    );
  });

  it("rejects a collapsed distribution that does not add up to one", () => {
    expect(
      validateAnswer("EI", choiceAnswer({ probabilities: { E: 0, I: 0 } }), options),
    ).toContain("合計");
  });

  it("rejects a choice that is not one of the question's options", () => {
    expect(validateAnswer("EI", choiceAnswer({ choice: "Z" }), options)).toContain("候補にない答え");
  });

  it("checks the noul value range", () => {
    const noul = {
      type: "noul",
      noul: Number.NaN,
      confidence: 0.5,
      answer_confidence: 0.5,
      action: { act_probability: 0.5 },
    } as NoulAnswer;
    expect(validateAnswer("EI", noul, options)).toContain("NaN");
    expect(validateAnswer("EI", { ...noul, noul: 1.5 } as NoulAnswer, options)).toContain(
      "0〜1",
    );
  });
});

describe("Agent.predictBatch output guards", () => {
  it("decodes a healthy pass into finite answers", async () => {
    const agent = batchAgent(() => healthyRows(2));
    const results = await agent.predictBatch(["一言目", "二行目"], questions);

    expect(results).toHaveLength(2);
    for (const result of results) {
      for (const id of ids) {
        const answer = result.answers[id] as ChoiceAnswer;
        expect(answer.type).toBe("choice");
        expect(Number.isFinite(answer.confidence)).toBe(true);
        for (const value of Object.values(answer.probabilities)) {
          expect(Number.isFinite(value)).toBe(true);
        }
      }
    }
    // The first question of each state leans positive, the second negative.
    expect((results[0].answers[ids[0]] as ChoiceAnswer).probabilities).toEqual({ E: 0.982, I: 0.018 });
  });

  it("throws instead of returning NaN probabilities", async () => {
    const agent = batchAgent(() => {
      const rows = healthyRows(1);
      rows.logits[2] = [Number.NaN, 0];
      return rows;
    });

    await expect(agent.predictBatch(["テスト"], questions)).rejects.toThrow(/NaN/);
  });

  it("throws on a truncated pass", async () => {
    const agent = batchAgent(() => ({ logits: [[1, 2]], act: [[1, 0]] }));
    await expect(agent.predictBatch(["テスト"], questions)).rejects.toThrow(
      /出力行が不足しています/,
    );
  });

  it("keeps working when a chunk ends mid-batch", async () => {
    // 3 states at 2 rows per state is 6 rows, above the 4-row batch size, so the
    // loop has to validate one pass and then a second, shorter one.
    const agent = batchAgent(() => healthyRows(2));
    const results = await agent.predictBatch(["a", "b", "c"], questions, {
      batchSize: 4,
    });

    expect(results.map((result) => result.model)).toEqual([
      "laya-rl-agent",
      "laya-rl-agent",
      "laya-rl-agent",
    ]);
    expect(results[2].answers[ids[3]]).toBeDefined();
  });

  it("does not spin on a non-finite batch size", async () => {
    const agent = batchAgent(() => healthyRows(1));
    await expect(
      agent.predictBatch(["テスト"], questions, { batchSize: Number.NaN }),
    ).resolves.toHaveLength(1);
  });

  it("rejects a broken question definition before running the model", async () => {
    const agent = batchAgent(() => healthyRows(1));
    const broken: Record<string, QuestionDef> = { EI: { type: "choice" } };
    await expect(agent.predictBatch(["テスト"], broken)).rejects.toThrow(/instructions/);
  });
});
