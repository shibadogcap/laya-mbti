import {
  TEMP_MAX,
  TEMP_MIN,
  buildQuestionPrefix,
  clampTemperature,
  collateItems,
  confidenceFromProbs,
  answerConfidence,
  renderOptions,
  sequenceWithState,
  serializeState,
  softmax,
  tempBucket,
} from "./common.js";
import type { Batch, SessionProvider } from "./providers.js";
import { encodeWithData, parseTokenizerJson, type TokenizerLike } from "./tokenizer.js";
import { decide, type DecideOptions, type DecisionResult } from "./structured.js";
import {
  HookRegistry,
  PredictContext,
  aggregateUsage,
  composeHooks,
  dispatch,
  normaliseHooks,
  type HookArg,
  type PredictHook,
} from "./hooks.js";

export const QTYPES: Record<string, number> = { choice: 0, score: 1, noul: 2 };

export interface QuestionDef {
  type: string;
  instructions?: unknown;
  criteria?: unknown;
  [k: string]: unknown;
}

export interface ActionInfo {
  act_probability: number;
}

export interface ChoiceAnswer {
  type: "choice";
  choice: string;
  probabilities: Record<string, number>;
  confidence: number;
  answer_confidence: number;
  action: ActionInfo;
}

export interface ScoreAnswer {
  type: "score";
  score: number;
  legend: Record<string, unknown>;
  probabilities: Record<string, number>;
  confidence: number;
  answer_confidence: number;
  action: ActionInfo;
}

export interface NoulAnswer {
  type: "noul";
  noul: number;
  confidence: number;
  answer_confidence: number;
  action: ActionInfo;
}

export type SystemAnswer = ChoiceAnswer | ScoreAnswer | NoulAnswer;

export interface SystemUsage {
  input_tokens: number;
  output_tokens: number;
}

export interface SystemOneResult {
  model: string;
  answers: Record<string, SystemAnswer>;
  usage: SystemUsage;
}

export interface AgentCfg {
  max_len?: number;
  head_max_len?: number;
  temperature?: unknown;
  temperature_by_options?: Record<string, unknown>;
  [k: string]: unknown;
}

export interface AgentOptions {
  provider: SessionProvider;
  tok?: TokenizerLike;
  cfg?: AgentCfg;
  max_len?: number;
  head_max_len?: number;
  temperature?: unknown;
  temperature_by_options?: Record<string, unknown>;
  /**
   * Per-language temperature overrides, keyed by language code; keys are normalised to
   * their base subtag (`de-AT` -> `de`), matching Python `Agent(lang_temperatures=...)`.
   * Each entry may carry a `temperature` list of 3 floats (default: the base raw
   * temperature) and/or a `temperature_by_options` map (default: none). A matching
   * override replaces the scale wholesale — see the note at the decode site.
   */
  lang_temperatures?: Record<
    string,
    { temperature?: unknown; temperature_by_options?: Record<string, unknown> } | null
  >;
  hooks?: HookArg;
  onPredictStart?: PredictHook;
  onPredictEnd?: PredictHook;
  hooksRaise?: boolean;
}

/** Per-call options shared by Agent.systemOne/predict and Router.predict. */
export interface PredictOptions {
  /**
   * Language of the request (e.g. "de"); when the Agent has a matching
   * `lang_temperatures` override it selects that language's temperature, exactly like
   * Python `system_one(..., lang=...)`. Routing alone never sets this.
   */
  lang?: string | null;
  hooks?: HookArg;
  onPredictStart?: PredictHook;
  onPredictEnd?: PredictHook;
  hooksRaise?: boolean;
}

function qidStr(qid: string): string {
  return JSON.stringify(qid);
}

export function checkQuestion(qid: string, qdef: unknown): void {
  if (typeof qdef !== "object" || qdef === null || Array.isArray(qdef)) {
    const got = Array.isArray(qdef) ? "list" : qdef === null ? "NoneType" : typeof qdef;
    throw new Error(`question ${qidStr(qid)}: definition must be a dict, got ${got}`);
  }
  const q = qdef as Record<string, unknown>;
  const t = q["type"];
  if (t !== "choice" && t !== "score" && t !== "noul") {
    throw new Error(
      `question ${qidStr(qid)}: unknown type ${JSON.stringify(t)}; use one of ${JSON.stringify(Object.keys(QTYPES).sort())}`,
    );
  }
  if (!("instructions" in q)) {
    throw new Error(`question ${qidStr(qid)}: no 'instructions'; add the text the model should answer`);
  }
  const crit = q["criteria"];
  if (t === "choice") {
    if (typeof crit !== "object" || crit === null) {
      throw new Error(
        `question ${qidStr(qid)}: a choice question takes 'criteria' as a dict of label -> description, or a list of labels`,
      );
    }
    if (Object.keys(crit as object).length === 0) {
      throw new Error(`question ${qidStr(qid)}: a choice question needs at least one criterion`);
    }
  } else if (t === "score") {
    if (!Array.isArray(crit)) {
      throw new Error(
        `question ${qidStr(qid)}: a score question takes 'criteria' as a list of level descriptions, index 0 first`,
      );
    }
    if (crit.length === 0) {
      throw new Error(`question ${qidStr(qid)}: a score question needs at least one level`);
    }
    const nullAt = crit.findIndex((c) => c === null || c === undefined);
    if (nullAt >= 0) {
      throw new Error(
        `question ${qidStr(qid)}: score level ${nullAt} is null; give every level a description, index 0 first`,
      );
    }
  } else if (crit !== undefined && crit !== null && (typeof crit !== "object" || Array.isArray(crit))) {
    throw new Error(
      `question ${qidStr(qid)}: a noul question takes 'criteria' as a dict with optional 'true'/'false' descriptions, or omits it`,
    );
  } else if (crit && typeof crit === "object" && !Array.isArray(crit)) {
    const invalid = Object.keys(crit as Record<string, unknown>).filter((key) => key !== "true" && key !== "false");
    if (invalid.length > 0) {
      throw new Error(
        `question ${qidStr(qid)}: noul criteria may contain only 'true' and 'false'; got ${JSON.stringify(invalid)}`,
      );
    }
  }
  if ("labels" in q && t !== "noul") {
    throw new Error(`question ${qidStr(qid)}: 'labels' is only supported for noul questions`);
  }
  if (t === "noul" && "labels" in q) {
    const labels = q["labels"];
    if (typeof labels !== "object" || labels === null || Array.isArray(labels)) {
      throw new Error(`question ${qidStr(qid)}: noul labels must be an object with 'false' and 'true'`);
    }
    const entries = Object.entries(labels as Record<string, unknown>);
    const keys = entries.map(([key]) => key).sort();
    if (keys.length !== 2 || keys[0] !== "false" || keys[1] !== "true" ||
        entries.some(([, value]) => typeof value !== "string" || value.trim() === "")) {
      throw new Error(`question ${qidStr(qid)}: noul labels must map exactly 'false' and 'true' to distinct non-empty strings`);
    }
    const falseLabel = String((labels as Record<string, unknown>)["false"]).trim();
    const trueLabel = String((labels as Record<string, unknown>)["true"]).trim();
    if (falseLabel === trueLabel) {
      throw new Error(`question ${qidStr(qid)}: noul labels must be distinct`);
    }
  }
}

export function toInternal(qdef: QuestionDef): { t: "choice" | "score" | "noul"; ins: string; crit: unknown; labels?: { false: string; true: string } } {
  const t = qdef["type"] as "choice" | "score" | "noul";
  let crit: unknown = qdef["criteria"];
  if (t === "choice" && Array.isArray(crit)) {
    crit = Object.fromEntries(crit.map((c: unknown) => [c as string, null]));
  } else if (t === "noul" && crit !== null && crit !== undefined && typeof crit === "object" && !Array.isArray(crit)) {
    crit = Object.fromEntries(Object.entries(crit as Record<string, unknown>).map(([k, v]) => [String(k).toLowerCase(), v]));
  }
  let ins: unknown = qdef["instructions"];
  if (typeof ins !== "string") ins = serializeState(ins);
  const out: { t: "choice" | "score" | "noul"; ins: string; crit: unknown; labels?: { false: string; true: string } } = {
    t, ins: ins as string, crit,
  };
  if (t === "noul" && qdef["labels"] && typeof qdef["labels"] === "object" && !Array.isArray(qdef["labels"])) {
    const labels = qdef["labels"] as Record<string, unknown>;
    out.labels = { false: String(labels["false"]).trim(), true: String(labels["true"]).trim() };
  }
  return out;
}

export function defaultTokenizer(): TokenizerLike {
  return {
    clsId: 101,
    sepId: 102,
    maskId: 103,
    padId: 0,
    maskToken: "[MASK]",
    encode(text: string): number[] {
      return text
        .split(/\s+/)
        .filter(Boolean)
        .map((w, i) => 1000 + ((w.length * 31 + i * 7) % 20000));
    },
  };
}

function tokenizerFromHF(tokenizerJson: unknown): TokenizerLike | null {
  const data = parseTokenizerJson(tokenizerJson);
  if (!data) return null;
  return {
    clsId: data.ids.cls,
    sepId: data.ids.sep,
    maskId: data.ids.mask,
    padId: data.ids.pad,
    maskToken: data.maskToken,
    encode: (text: string) => encodeWithData(data, text),
  };
}

const r4 = (v: number): number => Math.round(v * 1e4) / 1e4;

/** One head output row: the question it answers and how many options it must cover. */
export interface HeadRow {
  id: string;
  options: number;
}

const RETRY_HINT = "もう一度診断をリトライしてください。";

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function describeNumber(value: unknown): string {
  if (typeof value !== "number") return `数値ではない値（${typeof value}）`;
  if (Number.isNaN(value)) return "NaN";
  return value > 0 ? "+Infinity" : "-Infinity";
}

/**
 * Pure guard over one head output. Returns `null` when every row can be decoded
 * into an answer, otherwise a Japanese message naming the first unusable row.
 *
 * Without this a truncated or corrupted pass reaches `_answerFor`, whose softmax
 * then yields NaN probabilities and confidence, and those NaNs surface as a
 * confidently wrong MBTI type instead of an error. A partial softmax also looks
 * legitimate: `softmax([])` is empty and `softmax([0])` is `[1]`, so both produce
 * answers rather than throwing.
 */
export function validateHeadOutput(
  logits: unknown,
  act: unknown,
  rows: HeadRow[],
): string | null {
  if (!Array.isArray(logits) || !Array.isArray(act)) {
    return `モデルの出力が壊れています（logits / act が数値の配列ではありません）。${RETRY_HINT}`;
  }
  if (logits.length < rows.length || act.length < rows.length) {
    return (
      `モデルの出力行が不足しています（必要な ${rows.length} 行に対して logits ${logits.length} 行、act ${act.length} 行）。` +
      `${RETRY_HINT}`
    );
  }
  for (let r = 0; r < rows.length; r++) {
    const { id, options } = rows[r];
    const logitRow = logits[r];
    const actRow = act[r];
    if (!Array.isArray(logitRow) || !Array.isArray(actRow)) {
      return `モデルが質問「${id}」（行 ${r}）の答えを数値の配列として返しませんでした。${RETRY_HINT}`;
    }
    if (logitRow.length < options) {
      return (
        `モデルが質問「${id}」（行 ${r}）の答えを ${options} 個返すべきところ ${logitRow.length} 個しか返しませんでした。` +
        `${RETRY_HINT}`
      );
    }
    // `_answerFor` turns the first `options` logits into the answer distribution and
    // the whole act row into the escalation probability; both must be finite.
    for (let i = 0; i < options; i++) {
      if (!isFiniteNumber(logitRow[i])) {
        return `モデルが質問「${id}」（行 ${r}）に ${describeNumber(logitRow[i])} を返しました。答えを計算できないため、集計しません。${RETRY_HINT}`;
      }
    }
    if (actRow.length < 2) {
      return `モデルが質問「${id}」（行 ${r}）の確率を返しませんでした（act が ${actRow.length} 個しかありません）。${RETRY_HINT}`;
    }
    for (let i = 0; i < actRow.length; i++) {
      if (!isFiniteNumber(actRow[i])) {
        return `モデルが質問「${id}」（行 ${r}）の確率に ${describeNumber(actRow[i])} が含まれています。答えを計算できないため、集計しません。${RETRY_HINT}`;
      }
    }
  }
  return null;
}

/**
 * Pure guard over one decoded answer: the probabilities must cover the question's
 * options, add up to one, and the reported confidence must be a real number.
 * Catches a softmax that silently collapsed (all-zero or overflowing logits).
 */
export function validateAnswer(
  id: string,
  answer: SystemAnswer | undefined,
  options: number,
): string | null {
  if (typeof answer !== "object" || answer === null) {
    return `モデルが質問「${id}」の答えを返しませんでした。${RETRY_HINT}`;
  }
  if (answer.type === "noul") {
    if (!isFiniteNumber(answer.noul) || answer.noul < 0 || answer.noul > 1) {
      return `モデルが質問「${id}」の答えが ${describeNumber(answer.noul)} です。0〜1 の値が必要です。${RETRY_HINT}`;
    }
  } else {
    const values = Object.values(answer.probabilities ?? {});
    if (values.length !== options) {
      return (
        `モデルが質問「${id}」の確率を ${options} 個返すべきところ ${values.length} 個返しました。` +
        `${RETRY_HINT}`
      );
    }
    for (const value of values) {
      if (!isFiniteNumber(value)) {
        return `モデルが質問「${id}」の確率に ${describeNumber(value)} が含まれています。集計しません。${RETRY_HINT}`;
      }
    }
    const sum = values.reduce((a, b) => a + b, 0);
    // Each probability is rounded to 4 decimals, so the sum drifts by up to
    // 5e-5 per option; allow that much and no more.
    if (sum <= 0 || Math.abs(sum - 1) > Math.max(1e-3, 5e-5 * options)) {
      return `モデルが質問「${id}」の確率の合計が ${r4(sum)} です（1 になる必要があります）。集計しません。${RETRY_HINT}`;
    }
    if (answer.type === "choice" && !Object.hasOwn(answer.probabilities, answer.choice)) {
      return `モデルが質問「${id}」で、候補にない答え「${String(answer.choice)}」を返しました。${RETRY_HINT}`;
    }
    if (answer.type === "score" && !isFiniteNumber(answer.score)) {
      return `モデルが質問「${id}」のスコアが ${describeNumber(answer.score)} です。集計しません。${RETRY_HINT}`;
    }
  }
  if (!isFiniteNumber(answer.confidence)) {
    return `モデルが質問「${id}」の確信度が ${describeNumber(answer.confidence)} です。集計しません。${RETRY_HINT}`;
  }
  return null;
}

export class Agent extends HookRegistry {
  hooksRaise: boolean;
  cfg: AgentCfg;
  provider: SessionProvider;
  tok: TokenizerLike;
  maxLen: number;
  headMaxLen: number;
  private readonly questionPrefixCache = new Map<
    string,
    ReturnType<typeof buildQuestionPrefix>
  >();
  temperatureRaw: unknown;
  temperatureByOptionsRaw: Record<string, unknown>;
  temperature: number[];
  temperatureByOptions: Record<string, number>;
  langTemperatures: Record<
    string,
    { temperature: number[]; temperatureByOptions: Record<string, number> }
  >;

  constructor(opts: AgentOptions) {
    super();
    if (!opts || !opts.provider) throw new Error("Agent needs a provider");
    this.provider = opts.provider;
    // Hooks are opt-in; an unset hook list is a no-op. See hooks.ts.
    this.hooks = normaliseHooks(opts.hooks, opts.onPredictStart, opts.onPredictEnd);
    this.hooksRaise = opts.hooksRaise ?? true;
    const cfg = { ...(opts.cfg ?? {}) } as AgentCfg;
    if (opts.max_len !== undefined) cfg.max_len = opts.max_len;
    if (opts.head_max_len !== undefined) cfg.head_max_len = opts.head_max_len;
    if (opts.temperature !== undefined) cfg.temperature = opts.temperature;
    if (opts.temperature_by_options !== undefined) cfg.temperature_by_options = opts.temperature_by_options;
    this.cfg = cfg;
    this.maxLen = Number(cfg.max_len ?? 512);
    this.headMaxLen = Number(cfg.head_max_len ?? 192);
    this.tok = opts.tok ?? defaultTokenizer();
    const raw = (cfg.temperature ?? [1.0, 1.0, 1.0]) as unknown;
    this.temperatureRaw = raw;
    const rawList = Array.isArray(raw) ? raw : [raw, raw, raw];
    this.temperature = [0, 1, 2].map((i) => clampTemperature(rawList[i] ?? 1.0));
    this.temperatureByOptionsRaw = (cfg.temperature_by_options ?? {}) as Record<string, unknown>;
    this.temperatureByOptions = Object.fromEntries(
      Object.entries(this.temperatureByOptionsRaw).map(([k, v]) => [k, clampTemperature(v)]),
    );
    const entries: Array<[string, unknown, number]> = [
      ...Object.entries(this.temperatureByOptionsRaw).map(
        ([k, v]) => [k, v, this.temperatureByOptions[k]] as [string, unknown, number],
      ),
      ...[0, 1, 2].map(
        (i) => [`temperature[${i}]`, rawList[i] ?? 1.0, this.temperature[i]] as [string, unknown, number],
      ),
    ];
    const rejected: string[] = [];
    for (const [name, rawV, applied] of entries) {
      if (Number(rawV) === applied) continue;
      rejected.push(`${name}=${JSON.stringify(rawV) ?? String(rawV)} -> ${applied}`);
    }
    if (rejected.length > 0) {
      console.warn(
        `laya: this checkpoint ships invalid temperatures or values outside [${TEMP_MIN}, ${TEMP_MAX}]; ` +
          `using ${rejected.join(", ")}. Treat confidence from the affected entries as uncalibrated.`,
      );
    }
    // Mirrors Agent.__init__ (agent.py): keys normalise to the base subtag, an omitted
    // temperature defaults to the base raw temperature, and every value is clamped.
    this.langTemperatures = {};
    for (const [l, lc] of Object.entries(opts.lang_temperatures ?? {})) {
      const normL = l.split("-")[0].toLowerCase();
      const tRaw = lc?.temperature ?? rawList;
      if (!Array.isArray(tRaw) || tRaw.length !== 3) {
        throw new Error(
          `Language override ${JSON.stringify(l)} temperature must be a list of 3 floats`,
        );
      }
      const tboRaw = (lc?.temperature_by_options ?? {}) as Record<string, unknown>;
      this.langTemperatures[normL] = {
        temperature: [0, 1, 2].map((i) => clampTemperature(tRaw[i])),
        temperatureByOptions: Object.fromEntries(
          Object.entries(tboRaw).map(([k, v]) => [k, clampTemperature(v)]),
        ),
      };
    }
  }

  /**
   * Evaluate typed questions across one state in a single forward pass.
   *
   * `hooks` / `onPredictStart` / `onPredictEnd` observe or shape the prediction, appended
   * after any hooks installed on the Agent; a start hook may rewrite the state/questions or
   * call `ctx.skip(...)` to short-circuit inference, an end hook may rewrite the results.
   * See hooks.ts. `hooksRaise` overrides the Agent's setting for this call.
   */
  async systemOne(
    state: unknown,
    questions: Record<string, QuestionDef>,
    opts: PredictOptions = {},
  ): Promise<SystemOneResult> {
    return (await this._predictHooked([state], questions, opts))[0];
  }

  private async _predictHooked(
    states: unknown[],
    questions: Record<string, QuestionDef>,
    opts: PredictOptions,
  ): Promise<SystemOneResult[]> {
    const active = composeHooks(this.hooks, opts.hooks, opts.onPredictStart, opts.onPredictEnd);
    const raiseErrors = opts.hooksRaise ?? this.hooksRaise;
    const ctx = new PredictContext({
      states,
      questions: questions as Record<string, unknown>,
      agent: this,
    });
    try {
      dispatch(active, "onPredictStart", ctx, { raiseErrors });
      if (ctx.results === null) {
        const out: SystemOneResult[] = [];
        for (const st of ctx.states) {
          out.push(
            await this._systemOneCore(
              st,
              ctx.questions as Record<string, QuestionDef>,
              opts.lang ?? null,
            ),
          );
        }
        ctx.results = out as unknown as Record<string, unknown>[];
        ctx.model ??= out[0]?.model ?? null;
      }
    } catch (err) {
      ctx.error = err;
      try {
        dispatch(active, "onError", ctx, { raiseErrors });
      } catch {
        // A failing onError hook must not hide the failure that triggered it.
      }
      throw err;
    } finally {
      ctx.markElapsed();
      if (ctx.results !== null) ctx.usage = aggregateUsage(ctx.results);
      try {
        dispatch(active, "onPredictEnd", ctx, { raiseErrors });
      } catch (hookErr) {
        // End hooks run on the failure path too; do not let one mask the real error.
        if (ctx.error === null) throw hookErr;
      }
    }
    return ctx.results as unknown as SystemOneResult[];
  }

  private async _systemOneCore(
    state: unknown,
    questions: Record<string, QuestionDef>,
    lang: string | null = null,
  ): Promise<SystemOneResult> {
    const ids = Object.keys(questions ?? {});
    if (ids.length === 0) {
      return { model: "laya-rl-agent", answers: {}, usage: { input_tokens: 0, output_tokens: 0 } };
    }
    const internals: { t: "choice" | "score" | "noul"; ins: string; crit: unknown; labels?: { false: string; true: string } }[] = [];
    for (const qid of ids) {
      checkQuestion(qid, questions[qid]);
      internals.push(toInternal(questions[qid]));
    }
    const items = this._encodeStateItems(state, ids, internals);
    const collated = collateItems([items], this.tok.padId);
    if (!collated) throw new Error("no items to collate");
    const batch: Batch = collated;
    const nTokens = batch.attentionMask.flat().reduce((a, b) => a + b, 0);
    const { lastHidden } = await this.provider.runEncoder(batch);
    const { logits, act } = await this.provider.runHead(lastHidden, batch);
    const broken = validateHeadOutput(
      logits,
      act,
      items.map((item, r) => ({ id: ids[r], options: item.markers.length })),
    );
    if (broken) throw new Error(broken);

    const answers: Record<string, SystemAnswer> = {};
    for (let r = 0; r < ids.length; r++) {
      const answer = this._answerFor(
        internals[r],
        items[r].markers.length,
        logits[r] as number[],
        act[r] as number[],
        lang,
      );
      const unusable = validateAnswer(ids[r], answer, items[r].markers.length);
      if (unusable) throw new Error(unusable);
      answers[ids[r]] = answer;
    }
    return { model: "laya-rl-agent", answers, usage: { input_tokens: nTokens, output_tokens: 0 } };
  }

  /** Turns one row of head logits into a typed answer, applying temperature calibration. */
  private _answerFor(
    q: { t: "choice" | "score" | "noul"; ins: string; crit: unknown; labels?: { false: string; true: string } },
    k: number,
    logitsRow: number[],
    actRow: number[],
    lang: string | null,
  ): SystemAnswer {
    const qt = QTYPES[q.t];
    const bucket = tempBucket(qt, k);
    const langCfg = lang ? this.langTemperatures[lang.split("-")[0].toLowerCase()] : undefined;
    const scale = langCfg
      ? (langCfg.temperatureByOptions[bucket] ?? langCfg.temperature[qt] ?? 1.0)
      : (this.temperatureByOptions[bucket] ?? this.temperature[qt] ?? 1.0);
    const z = logitsRow.slice(0, k).map((v) => v / scale);
    const p = softmax(z);
    const actP = softmax((actRow ?? [1, 0]).slice(0, Math.max(2, actRow.length)));
    const ext = { act_probability: r4(actP[0]) };
    const ansConf = r4(answerConfidence(p));
    if (q.t === "choice") {
      const keys = Object.keys(q.crit as Record<string, unknown>);
      let best = 0;
      for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
      return {
        type: "choice",
        choice: keys[best],
        probabilities: Object.fromEntries(keys.map((kk, i) => [kk, r4(p[i] ?? 0)])),
        confidence: r4(confidenceFromProbs(p)),
        answer_confidence: ansConf,
        action: ext,
      };
    }
    if (q.t === "score") {
      const exp = p.reduce((a, v, i) => a + i * v, 0);
      return {
        type: "score",
        score: r4(exp),
        legend: Object.fromEntries((q.crit as unknown[]).map((c, i) => [String(i), c])),
        probabilities: Object.fromEntries(p.map((v, i) => [String(i), r4(v)])),
        confidence: r4(confidenceFromProbs(p)),
        answer_confidence: ansConf,
        action: ext,
      };
    }
    const pt = p[1] ?? 0;
    return {
      type: "noul",
      noul: r4(pt),
      confidence: r4(Math.max(pt, 1 - pt)),
      answer_confidence: ansConf,
      action: ext,
    };
  }

  /**
   * Evaluate the same questions over many states, packing their question rows into
   * shared forward passes. The exported graph's dynamic batch dimension caps rows at 64;
   * `batchSize` (in rows) can lower it. Results keep input order.
   *
   * Every pass is checked before it is decoded (`validateHeadOutput`), and every
   * decoded answer before it is returned (`validateAnswer`), so a truncated or
   * non-finite pass raises a Japanese error instead of returning NaN probabilities
   * that would later look like a confident result.
   */
  async predictBatch(
    states: unknown[],
    questions: Record<string, QuestionDef>,
    opts: PredictOptions & { batchSize?: number } = {},
  ): Promise<SystemOneResult[]> {
    if (states.length === 0) return [];
    const ids = Object.keys(questions ?? {});
    if (ids.length === 0) {
      return states.map(() => ({
        model: "laya-rl-agent",
        answers: {},
        usage: { input_tokens: 0, output_tokens: 0 },
      }));
    }
    for (const qid of ids) checkQuestion(qid, questions[qid]);
    const internals = ids.map((qid) => toInternal(questions[qid]));
    const lang = opts.lang ?? null;
    const requestedRows = Number(opts.batchSize);
    // A non-finite or non-positive batch size must not turn the chunk loop into a
    // NaN step, which would never advance.
    const maxRows = Math.max(
      ids.length,
      Math.min(
        64,
        Number.isFinite(requestedRows) ? Math.floor(requestedRows) : 64,
      ),
    );
    const statesPerChunk = Math.max(1, Math.floor(maxRows / ids.length));

    const results: SystemOneResult[] = new Array(states.length);
    for (let start = 0; start < states.length; start += statesPerChunk) {
      const chunk = states.slice(start, start + statesPerChunk);
      const perState = chunk.map((st) => this._encodeStateItems(st, ids, internals));
      const collated = collateItems(perState, this.tok.padId);
      if (!collated) throw new Error("no items to collate");
      const nTokens = collated.attentionMask.flat().reduce((a, b) => a + b, 0);
      const { lastHidden } = await this.provider.runEncoder(collated);
      const { logits, act } = await this.provider.runHead(lastHidden, collated);
      const rows: HeadRow[] = [];
      for (let s = 0; s < perState.length; s++) {
        for (let j = 0; j < ids.length; j++) {
          rows.push({ id: ids[j], options: perState[s][j].markers.length });
        }
      }
      const broken = validateHeadOutput(logits, act, rows);
      if (broken) throw new Error(broken);
      chunk.forEach((_st, s) => {
        const answers: Record<string, SystemAnswer> = {};
        for (let j = 0; j < ids.length; j++) {
          const row = s * ids.length + j;
          const answer = this._answerFor(
            internals[j],
            perState[s][j].markers.length,
            logits[row] as number[],
            act[row] as number[],
            lang,
          );
          const unusable = validateAnswer(ids[j], answer, perState[s][j].markers.length);
          if (unusable) throw new Error(unusable);
          answers[ids[j]] = answer;
        }
        results[start + s] = {
          model: "laya-rl-agent",
          answers,
          usage: { input_tokens: Math.round(nTokens / chunk.length), output_tokens: 0 },
        };
      });
    }
    return results;
  }

  /** Encodes one state once, then composes each question's prefix onto it. */
  private _encodeStateItems(
    state: unknown,
    ids: string[],
    internals: { t: "choice" | "score" | "noul"; ins: string; crit: unknown; labels?: { false: string; true: string } }[],
  ): { ids: number[]; markers: number[]; qtype: number }[] {
    const stAll = this.tok.encode(serializeState(state).split(this.tok.maskToken).join(" "));
    const items: { ids: number[]; markers: number[]; qtype: number }[] = [];
    for (let i = 0; i < ids.length; i++) {
      const q = internals[i];
      const cacheKey = `${i}:${JSON.stringify(q)}:${this.maxLen}:${this.headMaxLen}`;
      let prefix = this.questionPrefixCache.get(cacheKey);
      if (!prefix) {
        prefix = buildQuestionPrefix(this.tok, q, this.maxLen, this.headMaxLen);
        this.questionPrefixCache.set(cacheKey, prefix);
      }
      const { ids: seq, markers } = sequenceWithState(
        prefix,
        stAll,
        this.tok.sepId,
        this.maxLen,
        Array.isArray(state),
      );
      if (markers.length !== renderOptions(q).length) {
        throw new Error(`question ${qidStr(ids[i])} options exceed head_max_len=${this.headMaxLen}`);
      }
      items.push({ ids: seq, markers, qtype: QTYPES[q.t] });
    }
    return items;
  }

  async predict(
    state: unknown,
    questions: Record<string, QuestionDef>,
    opts: PredictOptions = {},
  ): Promise<SystemOneResult> {
    return this.systemOne(state, questions, opts);
  }

  /**
   * Releases the provider's sessions. The WebGPU fallback replaces this Agent
   * with a WASM one, and the encoder is resident in GPU memory until the session
   * is released, so the caller has to free it before building the replacement.
   */
  async dispose(): Promise<void> {
    await this.provider.dispose?.();
  }

  /**
   * Answer `state` against a JSON schema (or explicit `opts.questions`) and return typed
   * values — see `structured.ts`. Pass exactly one of `schema` or `opts.questions`; other
   * options are forwarded to `predict`.
   */
  async decide(
    state: unknown,
    schema: unknown,
    opts: DecideOptions & PredictOptions & { returnDetails: true },
  ): Promise<DecisionResult>;
  async decide(
    state: unknown,
    schema?: unknown,
    opts?: DecideOptions & PredictOptions,
  ): Promise<Record<string, unknown>>;
  async decide(
    state: unknown,
    schema?: unknown,
    opts: DecideOptions & PredictOptions = {},
  ): Promise<Record<string, unknown> | DecisionResult> {
    return decide(this, state, schema, opts);
  }

  static async load(
    modelDirOrRepo: string,
    opts?: {
      subfolder?: string | null;
      numThreads?: number;
      /** Directory containing the ONNX Runtime `.wasm`/`.mjs` assets (self-hosted). */
      wasmPaths?: string;
      encoderProviders?: string[];
      headProviders?: string[];
      encoderGraphOptimizationLevel?: "disabled" | "basic" | "extended" | "layout" | "all";
      headGraphOptimizationLevel?: "disabled" | "basic" | "extended" | "layout" | "all";
      /** Per-language temperature overrides; see AgentOptions.lang_temperatures. */
      lang_temperatures?: AgentOptions["lang_temperatures"];
    },
  ): Promise<Agent> {
    const sub = opts?.subfolder ?? null;
    const { loadWebBundle, createWebProvider } = await import("./providers.js");
    const bundle = await loadWebBundle(modelDirOrRepo, { subfolder: sub });
    const cfg = bundle.cfg as AgentCfg;
    const tokenizerJson = bundle.tokenizerJson;
    const dir = bundle.dir;
    const provider = await createWebProvider(dir, {
      numThreads: opts?.numThreads,
      wasmPaths: opts?.wasmPaths,
      encoderProviders: opts?.encoderProviders,
      headProviders: opts?.headProviders,
      encoderGraphOptimizationLevel: opts?.encoderGraphOptimizationLevel,
      headGraphOptimizationLevel: opts?.headGraphOptimizationLevel,
    });
    try {
      if (!tokenizerJson) {
        throw new Error(
          `Incompatible model: tokenizer.json is missing or invalid in ${JSON.stringify(dir)}`,
        );
      }
      let tok: TokenizerLike;
      try {
        const parsed = tokenizerFromHF(tokenizerJson);
        if (!parsed) throw new Error("unsupported tokenizer.json format");
        tok = parsed;
      } catch (error) {
        throw new Error(
          `Incompatible model: tokenizer.json is missing or invalid in ${JSON.stringify(dir)}: ${String(error)}`,
        );
      }
      return new Agent({
        provider,
        tok,
        cfg,
        lang_temperatures: opts?.lang_temperatures,
      });
    } catch (error) {
      // The sessions are ~700 MB; an unusable tokenizer must not leave them
      // resident with no Agent pointing at them.
      await provider.dispose?.();
      throw error;
    }
  }
}
