import type { QuestionDef } from "../vendor/laya-ts/agent.js";
import type { AxisId, AxisMeta } from "./types.js";

export const AXES: AxisMeta[] = [
  {
    id: "EI",
    name: "外向 E / 内向 I",
    positive: {
      code: "E",
      label: "外向 (Extraversion)",
      description: "社交的で人との交流から活力を得る。",
    },
    negative: {
      code: "I",
      label: "内向 (Introversion)",
      description: "内省的で一人の時間を好む。",
    },
    neutral: "どちらとも言えない。",
  },
  {
    id: "SN",
    name: "感覚 S / 直観 N",
    positive: {
      code: "S",
      label: "感覚 (Sensing)",
      description: "具体的・現実的で事実や詳細を重視する。",
    },
    negative: {
      code: "N",
      label: "直観 (Intuition)",
      description: "抽象的・未来的で可能性や意味を重視する。",
    },
    neutral: "どちらとも言えない。",
  },
  {
    id: "TF",
    name: "思考 T / 感情 F",
    positive: {
      code: "T",
      label: "思考 (Thinking)",
      description: "論理と客観性で判断する。",
    },
    negative: {
      code: "F",
      label: "感情 (Feeling)",
      description: "共感と調和で判断する。",
    },
    neutral: "どちらとも言えない。",
  },
  {
    id: "JP",
    name: "判断 J / 知覚 P",
    positive: {
      code: "J",
      label: "判断 (Judging)",
      description: "計画・決定・秩序を好む。",
    },
    negative: {
      code: "P",
      label: "知覚 (Perceiving)",
      description: "柔軟・即興・開放性を好む。",
    },
    neutral: "どちらとも言えない。",
  },
];

/**
 * Builds the per-tweet question batch: one two-way `choice` per axis.
 *
 * A three-way `choice` (positive / negative / neutral) dilutes the probability
 * and collapses confidence, while two independent `noul` questions cost twice
 * as many forward passes for no accuracy gain. A two-way `choice` keeps the
 * margin between the poles and needs a single pass per axis.
 *
 * `order` and `flipOptions` exist to break the model's position bias. Asking
 * "E or I" first for every post, with the same option first, lets a preference
 * for the first row leak into the answer as a constant offset. Both are decided
 * once per run, so every post in a run is scored the same way and the aggregate
 * stays comparable, while two runs of the same archive do not share the bias.
 */
export function buildQuestions(
  order: readonly AxisId[] = AXES.map((axis) => axis.id),
  flipOptions: ReadonlySet<AxisId> = new Set(),
): Record<string, QuestionDef> {
  const questions: Record<string, QuestionDef> = {};
  for (const id of order) {
    const axis = AXES.find((candidate) => candidate.id === id);
    if (!axis) continue;
    const [first, second] = flipOptions.has(id)
      ? [axis.negative, axis.positive]
      : [axis.positive, axis.negative];
    questions[axis.id] = {
      type: "choice",
      instructions: `次のX(Twitter)の投稿の書き手の性格傾向を選んでください。観点: ${axis.name}`,
      criteria: {
        [first.code]: first.description,
        [second.code]: second.description,
      },
    } satisfies QuestionDef;
  }
  return questions;
}

/** Fisher-Yates over a copy, so callers keep their own order untouched. */
export function shuffle<T>(items: readonly T[], random: () => number = Math.random): T[] {
  const out = [...items];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

/** One run's question layout: axis order plus which axes list the pole first. */
export function randomizedQuestionPlan(random: () => number = Math.random): {
  order: AxisId[];
  flipOptions: Set<AxisId>;
} {
  const ids = AXES.map((axis) => axis.id);
  const order = shuffle(ids, random) as AxisId[];
  const flipOptions = new Set<AxisId>(ids.filter(() => random() < 0.5));
  return { order, flipOptions };
}
