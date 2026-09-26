import { createEffect, createMemo, createSignal, onCleanup, Show } from "solid-js";
import { DiagnosisProgress } from "./components/DiagnosisProgress.js";
import { DropZone } from "./components/DropZone.js";
import { PeriodPicker } from "./components/PeriodPicker.js";
import { ResultCard } from "./components/ResultCard.js";
import { TweetResults } from "./components/TweetResults.js";
import { aggregate } from "./lib/aggregate.js";
import { formatPercent, typeAgreement } from "./lib/agreement.js";
import { readArchive, type ArchiveProgress, type ArchiveTask } from "./lib/archive.js";
import { filterByPeriod, preFilter, sampleTweets } from "./lib/filter.js";
import {
  InferenceClient,
  type AnalyzeProgress,
  type AnalyzedItem,
  type EncoderBackend,
  type LoadProgressEvent,
} from "./lib/inference.js";
import { downloadResultImage as downloadResultImageFile } from "./lib/share.js";
import type { ArchiveSummary, MbtiResult } from "./lib/types.js";
import { formatBytes, safeImageSource } from "./lib/format.js";

type ModelState = "idle" | "loading" | "ready" | "error";
type ArchiveState = "idle" | "parsing" | "ready" | "error";
type AnalyzeState = "idle" | "running" | "done";

const errorMessage = (error: unknown) =>
  error instanceof Error ? error.message : String(error);

/** What the active backend costs the user; the specific cause comes from the worker. */
const BACKEND_NOTE: Record<EncoderBackend, string> = {
  webgpu: "",
  wasm:
    "WebGPU が使えない、または GPU で実行できなかったため、CPU で自動的にリトライしています。" +
    "WebGPU で実行する場合よりかなり時間がかかるので、解析する最大件数を絞ると早く終わります。",
};

function modelBaseUrl(): string {
  const override = import.meta.env.VITE_MODEL_URL as string | undefined;
  if (override) return override;
  return new URL("models/laya/", document.baseURI).href;
}

/**
 * The address shares point at. It is pinned rather than read from `location`, so
 * a share sent from the GitHub Pages host or a review tunnel still carries the
 * custom domain. `tests/share.test.ts` keeps it equal to the canonical tag in
 * `index.html`.
 */
export const SITE_URL = "https://laya-mbti.shibadogcap.com";

/** Public repository this project will live in, linked from the footer. */
export const REPO_URL = "https://github.com/shibadogcap/laya-mbti";
export const OWNER_GITHUB_URL = "https://github.com/shibadogcap";
export const OWNER_X_URL = "https://x.com/4ba_ba_baba";

function ortBaseUrl(): string {
  return new URL("ort/", document.baseURI).href;
}

export default function App() {
  const [modelState, setModelState] = createSignal<ModelState>("idle");
  const [modelProgress, setModelProgress] = createSignal<LoadProgressEvent | null>(null);
  const [modelError, setModelError] = createSignal<string | null>(null);
  /** Which execution provider is actually running the model, once known. */
  const [backend, setBackend] = createSignal<EncoderBackend | null>(null);
  /** Why the active backend is not WebGPU. */
  const [backendReason, setBackendReason] = createSignal<string | null>(null);

  const [archiveState, setArchiveState] = createSignal<ArchiveState>("idle");
  const [archive, setArchive] = createSignal<ArchiveSummary | null>(null);
  const [archiveProgress, setArchiveProgress] = createSignal<ArchiveProgress | null>(null);
  const [archiveError, setArchiveError] = createSignal<string | null>(null);
  const [analysisError, setAnalysisError] = createSignal<string | null>(null);
  const [localAvatarUrl, setLocalAvatarUrl] = createSignal<string | null>(null);
  let avatarObjectUrl: string | null = null;

  createEffect(() => {
    const blob = archive()?.account.avatarBlob;
    if (avatarObjectUrl) URL.revokeObjectURL(avatarObjectUrl);
    avatarObjectUrl = blob ? URL.createObjectURL(blob) : null;
    setLocalAvatarUrl(avatarObjectUrl);
  });

  onCleanup(() => {
    if (avatarObjectUrl) URL.revokeObjectURL(avatarObjectUrl);
  });

  const [minLength, setMinLength] = createSignal(15);
  const [includeReplies, setIncludeReplies] = createSignal(false);
  const [excludeRetweets, setExcludeRetweets] = createSignal(true);
  const [from, setFrom] = createSignal<number | null>(null);
  const [to, setTo] = createSignal<number | null>(null);
  const [marginThreshold, setMarginThreshold] = createSignal(0.1);
  const [maxTweets, setMaxTweets] = createSignal(500);

  const [analyzeState, setAnalyzeState] = createSignal<AnalyzeState>("idle");
  const [analyzeProgress, setAnalyzeProgress] = createSignal<AnalyzeProgress | null>(null);
  const [result, setResult] = createSignal<MbtiResult | null>(null);
  const [analysisWarning, setAnalysisWarning] = createSignal<string | null>(null);
  const [liveItem, setLiveItem] = createSignal<AnalyzedItem | null>(null);
  const [analyzedItems, setAnalyzedItems] = createSignal<AnalyzedItem[]>([]);
  const [shareNotice, setShareNotice] = createSignal<string | null>(null);

  let client: InferenceClient | null = null;
  let archiveTask: ArchiveTask | null = null;

  onCleanup(() => {
    client?.dispose();
    archiveTask?.cancel();
  });

  const tweets = () => archive()?.tweets ?? [];
  const minTs = createMemo(() => {
    const list = tweets();
    return list.length > 0 ? list[0].createdAt : 0;
  });
  const maxTs = createMemo(() => {
    const list = tweets();
    return list.length > 0 ? list[list.length - 1].createdAt : 0;
  });

  const periodTweets = createMemo(() => filterByPeriod(tweets(), from(), to()));
  const filtered = createMemo(() =>
    preFilter(periodTweets(), {
      minLength: minLength(),
      includeReplies: includeReplies(),
      excludeRetweets: excludeRetweets(),
    }),
  );
  const selected = createMemo(() => sampleTweets(filtered().kept, maxTweets()));

  const canAnalyze = () =>
    modelState() === "ready" &&
    archiveState() === "ready" &&
    analyzeState() !== "running" &&
    selected().length > 0;

  function ensureModel(): void {
    if (client) return;
    // A missing navigator.gpu is no longer a blocker: the worker falls back to
    // the CPU execution provider, and the status line explains the slowdown.
    const instance = new InferenceClient();
    client = instance;
    setModelState("loading");
    instance
      .load(modelBaseUrl(), ortBaseUrl(), setModelProgress)
      .then((info) => {
        setBackend(info.backend);
        setBackendReason(info.reason ?? null);
        setModelState("ready");
      })
      .catch((error: unknown) => {
        setModelError(errorMessage(error));
        setModelState("error");
      });
  }

  function reloadModel(): void {
    client?.dispose();
    client = null;
    setModelError(null);
    setModelProgress(null);
    setBackend(null);
    setBackendReason(null);
    setModelState("idle");
    ensureModel();
  }

  async function handleFiles(files: File[]): Promise<void> {
    if (files.length === 0) return;
    archiveTask?.cancel();
    archiveTask = null;
    setArchive(null);
    setResult(null);
    setAnalysisWarning(null);
    setLiveItem(null);
    setShareNotice(null);
    setArchiveError(null);
    setAnalysisError(null);
    setArchiveState("parsing");
    ensureModel();

    const task = readArchive(files, setArchiveProgress);
    archiveTask = task;
    try {
      const summary = await task.promise;
      setArchive(summary);
      setArchiveState("ready");
      const latest = summary.tweets.at(-1)?.createdAt ?? 0;
      setFrom(latest > 0 ? latest - 365 * 24 * 60 * 60 * 1000 : null);
      setTo(null);
    } catch (error) {
      if (errorMessage(error).includes("キャンセル")) return;
      setArchiveError(errorMessage(error));
      setArchiveState("error");
    }
  }

  async function startAnalysis(): Promise<void> {
    if (!client || !canAnalyze()) return;
    setResult(null);
    setAnalysisWarning(null);
    setAnalysisError(null);
    setLiveItem(null);
    setAnalyzedItems([]);
    setShareNotice(null);
    setAnalyzeProgress(null);
    setAnalyzeState("running");
    const payload = selected().map((tweet) => ({
      id: tweet.id,
      text: tweet.text,
    }));
    try {
      const items = await client.analyze(payload, {
        onProgress: (progress) => {
          if (progress.backend) setBackend(progress.backend);
          setAnalyzeProgress(progress);
        },
        onItem: (item) => setLiveItem(item),
        onRestart: (info) => {
          // The worker is starting over on the CPU: drop the WebGPU items it had
          // already sent and rewind the count, so nothing partial reads as done.
          setBackend(info.backend);
          setBackendReason(info.reason);
          setLiveItem(null);
          setAnalyzeProgress({
            done: 0,
            total: info.total,
            elapsedMs: 0,
            etaMs: 0,
            batch: 0,
            backend: info.backend,
          });
        },
      });
      const nextResult = aggregate(items, {
        marginThreshold: marginThreshold(),
      });
      if (nextResult.insufficientEvidence) {
        setResult(null);
        setAnalysisWarning(
          nextResult.invalidAnswers > 0
            ? "モデルの有効な回答を取得できませんでした。推論環境を確認して再試行してください。"
            : "しきい値に達する回答がありません。しきい値を下げるか、対象期間を広げてください。",
        );
        setLiveItem(null);
        setAnalyzeProgress(null);
        setAnalyzeState("idle");
        return;
      }
      setResult(nextResult);
      setAnalyzedItems(items);
      if (nextResult.invalidAnswers > 0) {
        setAnalysisWarning(
          `モデル出力の無効な回答を${nextResult.invalidAnswers.toLocaleString()}件除外しました。`,
        );
      }
      setLiveItem(null);
      setAnalyzeState("done");
      scrollToResult();
    } catch (error) {
      if (errorMessage(error).includes("キャンセル")) {
        setAnalyzeState("idle");
        setLiveItem(null);
        return;
      }
      setAnalysisError(errorMessage(error));
      setAnalyzeState("idle");
      setAnalyzeProgress(null);
      setLiveItem(null);
    }
  }

  /**
   * A finished run leaves the reader at the progress card, which is about to
   * disappear. Move to the result once it is in the DOM, and only when the user
   * is already near the top, so a re-run started from further down the page does
   * not yank the viewport.
   */
  function scrollToResult(): void {
    const target = document.querySelector(".result");
    if (!target) return;
    if (window.scrollY > window.innerHeight * 0.6) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    window.requestAnimationFrame(() => {
      target.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
    });
  }

  function cancelAnalysis(): void {
    client?.cancel();
    setAnalysisError(null);
    setAnalysisWarning(null);
    setAnalyzeState("idle");
    setAnalyzeProgress(null);
    setLiveItem(null);
    setAnalyzedItems([]);
  }

  function reset(): void {
    archiveTask?.cancel();
    setArchive(null);
    setArchiveProgress(null);
    setArchiveState("idle");
    setResult(null);
    setAnalysisWarning(null);
    setAnalysisError(null);
    setLiveItem(null);
    setShareNotice(null);
    setAnalyzeProgress(null);
    setAnalyzeState("idle");
    setAnalyzedItems([]);
    setArchiveError(null);
  }

  const modelPercent = () => {
    const progress = modelProgress();
    if (!progress || !progress.total || progress.total <= 0) return null;
    return Math.min(1, (progress.loaded ?? 0) / progress.total);
  };

  const archivePercent = () => {
    const progress = archiveProgress();
    if (!progress || progress.totalBytes <= 0) return null;
    return Math.min(1, progress.loaded / progress.totalBytes);
  };

  /** The one line that says which execution provider is running the model. */
  const backendNote = () => {
    const active = backend();
    if (!active) return null;
    const reason = backendReason();
    return [BACKEND_NOTE[active], reason].filter(Boolean).join(" ") || null;
  };

  const currentAnalysisProgress = () =>
    analyzeProgress() ?? {
      done: 0,
      total: selected().length,
      elapsedMs: 0,
      etaMs: 0,
      batch: 0,
      backend: backend() ?? undefined,
    };

  function shareResult(value: MbtiResult): void {
    const text =
      `${value.usedTweets.toLocaleString()}件の投稿を分析したら、` +
      `私のMBTIは ${value.type} でした！ #laya-mbti`;
    const intent = new URL("https://x.com/intent/post");
    intent.searchParams.set("text", `${text}\n${SITE_URL}`);
    intent.searchParams.set("url", SITE_URL);
    const opened = window.open(intent.href, "_blank", "noopener,noreferrer");
    setShareNotice(opened ? "Xの投稿画面を開きました" : "ポップアップを許可してください");
  }

  /** Same numbers the card shows, redrawn onto a canvas so it can be saved. */
  async function downloadResultImage(value: MbtiResult): Promise<void> {
    setShareNotice("画像を生成しています…");
    const alternatives = value.ranking
      .slice(1)
      .map((entry) => ({ entry, agreement: typeAgreement(entry) }))
      .sort((a, b) => b.agreement - a.agreement || a.entry.rank - b.entry.rank)
      .slice(0, 15)
      .map(({ entry, agreement }) => ({ type: entry.type, percent: formatPercent(agreement) }));
    try {
      await downloadResultImageFile({
        result: value,
        accountUsername: archive()?.account.username,
        avatarUrl: localAvatarUrl() ?? safeImageSource(archive()?.account.avatarUrl),
        agreement: typeAgreement(value.ranking[0]),
        alternatives,
      });
      setShareNotice("画像を保存しました");
    } catch (error) {
      setShareNotice(`画像を生成できませんでした: ${errorMessage(error)}`);
    }
  }

  return (
    <div class="app">
      <a class="skip-link" href="#main">本文へ移動</a>

      <main id="main">
        <section class="hero hero--single">
          <div class="hero__copy hero__copy--single">
            <h1 class="hero__title hero__title--wide">
              あなたの投稿を、<br />
              <span>4つの軸</span>で読み解く。
            </h1>
            <p class="hero__lead">
              AIが投稿を4つの軸から分析します。処理はブラウザ内で行い、投稿本文を送信しません。
            </p>
          </div>
          <div class="hero__visual">
            <img
              src="/hero-creator.png"
              alt="投稿アーカイブを抱えた人物と、投稿・人物・動画のアイコンが並ぶイラスト"
              width="400"
              height="400"
              decoding="async"
            />
          </div>
        </section>

        <div class="flow-strip" aria-label="診断の流れ">
          <div
            class="flow-step flow-step--primary"
            classList={{ "flow-step--current": archiveState() !== "ready" }}
            aria-current={archiveState() !== "ready" ? "step" : undefined}
          >
            <span>01</span>
            <strong>Xアーカイブを準備</strong>
          </div>
          <div
            class="flow-step flow-step--setup"
            classList={{
              "flow-step--current": archiveState() === "ready" && analyzeState() === "idle",
            }}
            aria-current={
              archiveState() === "ready" && analyzeState() === "idle" ? "step" : undefined
            }
          >
            <span>02</span>
            <strong>対象を決める</strong>
          </div>
          <div
            class="flow-step flow-step--diagnosis"
            classList={{ "flow-step--current": analyzeState() !== "idle" }}
            aria-current={analyzeState() !== "idle" ? "step" : undefined}
          >
            <span>03</span>
            <strong>傾向を読む</strong>
          </div>
        </div>

      <DropZone
        onFiles={(files) => void handleFiles(files)}
        busy={archiveState() === "parsing" || analyzeState() === "running"}
        hasArchive={archiveState() === "ready"}
      />

      <Show when={archiveState() === "parsing" && archiveProgress()}>
        {(progress) => (
          <section class="panel progress-panel">
            <header class="panel__header">
              <div>
                <h2>アーカイブを準備しています</h2>
              </div>
              <span class="panel__meta">
                ファイル {(progress().fileIndex ?? 0) + 1} / {progress().fileTotal}
              </span>
            </header>
            <p class="panel__line">{progress().message}</p>
            <Show when={archivePercent() !== null}>
              <div
                class="progress-track"
                role="progressbar"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round((archivePercent() ?? 0) * 100)}
                aria-label="アーカイブの読み込み進捗"
              >
                <div
                  class="progress-track__fill"
                  style={{ width: `${(archivePercent() ?? 0) * 100}%` }}
                />
              </div>
              <p class="panel__meta">
                {formatBytes(progress().loaded)} / {formatBytes(progress().totalBytes)}
              </p>
            </Show>
          </section>
        )}
      </Show>

      <Show when={archiveError()}>
        <p class="alert alert--error">{archiveError()}</p>
      </Show>

      <Show when={analysisError()}>
        <p class="alert alert--error analysis-error">
          推論を完了できませんでした: {analysisError()}
        </p>
      </Show>

      <Show when={archiveState() === "ready" && archive()}>
        {(summary) => (
          <section class="panel summary-panel summary-panel--archive">
            <header class="panel__header summary-panel__header">
              <h2>対象投稿を確認</h2>
              <button
                type="button"
                class="icon-button summary-panel__refresh"
                onClick={reset}
                aria-label="アーカイブを再読み込み"
              >
                <svg
                  viewBox="0 0 24 24"
                  width="18"
                  height="18"
                  fill="none"
                  stroke="currentColor"
                  stroke-width="1.7"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  aria-hidden="true"
                >
                  <path d="M20 11a8 8 0 0 0-14.7-4L4 9" />
                  <path d="M4 4v5h5" />
                  <path d="M4 13a8 8 0 0 0 14.7 4L20 15" />
                  <path d="M20 20v-5h-5" />
                </svg>
              </button>
            </header>
            <dl class="stats stats--summary">
              <Show when={summary().account.username}>
                <div class="stat stat--account">
                  <span class="stat__icon" aria-hidden="true">
                    <Show
                      when={localAvatarUrl() ?? safeImageSource(summary().account.avatarUrl)}
                      fallback={
                        <svg
                          viewBox="0 0 24 24"
                          width="18"
                          height="18"
                          fill="none"
                          stroke="currentColor"
                          stroke-width="1.7"
                          stroke-linecap="round"
                          stroke-linejoin="round"
                        >
                          <circle cx="12" cy="8" r="3" />
                          <path d="M5 20c.7-3.2 3.1-5 7-5s6.3 1.8 7 5" />
                        </svg>
                      }
                    >
                      {(avatarUrl) => <img class="stat__avatar" src={avatarUrl()} alt="" />}
                    </Show>
                  </span>
                  <dt class="stat__term">アカウント</dt>
                  <dd class="stat__value">@{summary().account.username}</dd>
                </div>
              </Show>
              <div class="stat stat--tweets">
                <span class="stat__icon" aria-hidden="true">
                  <svg
                    viewBox="0 0 24 24"
                    width="18"
                    height="18"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="1.7"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  >
                    <path d="M6 4.5h9l3 3v12H6z" />
                    <path d="M15 4.5v3h3M9 11h6M9 14h6" />
                  </svg>
                </span>
                <dt class="stat__term">投稿</dt>
                <dd class="stat__value numeric">{summary().tweets.length.toLocaleString()}</dd>
              </div>
              <div class="stat stat--period">
                <span class="stat__icon" aria-hidden="true">
                  <svg
                    viewBox="0 0 24 24"
                    width="18"
                    height="18"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="1.7"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  >
                    <rect x="4" y="5" width="16" height="15" rx="2" />
                    <path d="M8 3v4M16 3v4M4 9h16" />
                  </svg>
                </span>
                <dt class="stat__term">対象期間</dt>
                <dd class="stat__value">
                  {minTs() > 0
                    ? `${new Date(minTs()).toLocaleDateString("ja-JP")} 〜 ${new Date(
                        maxTs(),
                      ).toLocaleDateString("ja-JP")}`
                    : "不明"}
                </dd>
              </div>
            </dl>
          </section>
        )}
      </Show>

      <section class="setup-area" aria-label="診断の準備">
        <Show when={modelState() !== "idle"}>
          <section
            class="model-status model-status--compact"
            classList={{
              "model-status--ready": modelState() === "ready",
              "model-status--loading": modelState() === "loading",
              "model-status--error": modelState() === "error",
            }}
            role="status"
            aria-label="解析モデルの準備状況"
          >
            <div class="model-status__header">
              <div class="model-status__heading">
                <span class="model-status__mark" aria-hidden="true" />
                <h2>解析モデルの準備</h2>
              </div>


            </div>
            <Show when={modelState() === "loading"}>
              <p class="model-status__detail">
                {modelProgress()?.cached
                  ? `${modelProgress()?.file} をキャッシュから読み込み中`
                  : `${modelProgress()?.file ?? "モデル"} を読み込み中`}
                {modelPercent() !== null && !modelProgress()?.cached
                  ? ` (${Math.round((modelPercent() ?? 0) * 100)}%)`
                  : ""}
              </p>
              <div
                class="progress-track model-status__progress"
                role="progressbar"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round((modelPercent() ?? 0) * 100)}
                aria-label="解析モデルの読み込み進捗"
              >
                <div
                  class="progress-track__fill"
                  style={{ width: `${(modelPercent() ?? 0) * 100}%` }}
                />
              </div>
            </Show>
            <Show when={modelState() === "error" && modelError()}>
              <div class="model-status__error">
                <p class="alert alert--error">{modelError()}</p>
                <button
                  type="button"
                  class="chip model-status__retry"
                  onClick={reloadModel}
                >
                  リトライ
                </button>
              </div>
            </Show>
          </section>
        </Show>

        <Show when={archiveState() === "ready" && analyzeState() !== "running"}>
          <div class="setup-grid setup-grid--compact">
            <PeriodPicker
              minTs={minTs()}
              maxTs={maxTs()}
              from={from()}
              to={to()}
              setFrom={setFrom}
              setTo={setTo}
              count={periodTweets().length}
              total={tweets().length}
            />

            <section class="setup-panel setup-panel--settings">
              <header class="panel__header setup-panel__header">
                <h2>対象と分析方法</h2>
                <strong class="panel__meta setup-panel__count numeric">
                  選択中 {selected().length.toLocaleString()} /{" "}
                  {periodTweets().length.toLocaleString()} 件
                </strong>
              </header>
              <div class="settings setup-panel__settings">
                <label class="field">
                  <span>最低文字数（リンク等を除く）</span>
                  <input
                    type="number"
                    min="0"
                    max="200"
                    value={minLength()}
                    onInput={(event) =>
                      setMinLength(Math.max(0, Number(event.currentTarget.value) || 0))
                    }
                  />
                </label>
                <label class="field field--max">
                  <span>
                    解析する最大件数（期間全体から均等に抽出。0で全件）
                  </span>
                  <input
                    type="number"
                    min="0"
                    max="100000"
                    step="50"
                    value={maxTweets()}
                    onInput={(event) =>
                      setMaxTweets(Math.max(0, Number(event.currentTarget.value) || 0))
                    }
                  />
                </label>
                <label class="field field--check">
                  <input
                    type="checkbox"
                    checked={excludeRetweets()}
                    onChange={(event) => setExcludeRetweets(event.currentTarget.checked)}
                  />
                  <span>リポストを除外</span>
                </label>
                <label class="field field--check" title="他の人への返信と、自分自身への返信（スレッド）。本文が@で始まる投稿も対象">
                  <input
                    type="checkbox"
                    checked={includeReplies()}
                    onChange={(event) => setIncludeReplies(event.currentTarget.checked)}
                  />
                  <span>返信と自スレッドを含める</span>
                </label>
                <label class="field field--range">
                  <span>
                    有効票のしきい値（肯定と否定の確率差がこれ未満の票は保留扱い）:{" "}
                    {marginThreshold().toFixed(2)}
                  </span>
                  <input
                    type="range"
                    min="0"
                    max="0.5"
                    step="0.01"
                    value={marginThreshold()}
                    style={`--range-progress: ${(marginThreshold() / 0.5) * 100}%`}
                    onInput={(event) => setMarginThreshold(Number(event.currentTarget.value))}
                  />
                </label>
              </div>
              <p class="filter-stats__caption">前処理で除外した投稿の件数</p>
              <ul class="filter-stats setup-panel__filter-stats numeric">
                <li class="filter-stat">
                  <span class="filter-stat__label">短文</span>
                  <span class="filter-stat__count">{filtered().stats.tooShort}</span>
                </li>
                <li class="filter-stat">
                  <span class="filter-stat__label">リンクのみ</span>
                  <span class="filter-stat__count">{filtered().stats.linkOnly}</span>
                </li>
                <li class="filter-stat">
                  <span class="filter-stat__label" title="リポストとして記録された投稿">
                    リポスト
                  </span>
                  <span class="filter-stat__count">{filtered().stats.retweets}</span>
                </li>
                <li class="filter-stat">
                  <span
                    class="filter-stat__label"
                    title="他のアカウントへの返信と、自分自身への返信（スレッド）"
                  >
                    返信
                  </span>
                  <span class="filter-stat__count">{filtered().stats.replies}</span>
                </li>
                <li class="filter-stat">
                  <span class="filter-stat__label" title="本文が同一で重複した投稿">
                    重複投稿
                  </span>
                  <span class="filter-stat__count">{filtered().stats.duplicateTweets}</span>
                </li>
              </ul>
              <p class="filter-stats__kept">
                残った {filtered().stats.kept.toLocaleString()} 件から{" "}
                <Show
                  when={maxTweets() > 0 && maxTweets() < filtered().stats.kept}
                  fallback={<>全件を解析します</>}
                >
                  期間全体に均等にランダム抽出して {maxTweets().toLocaleString()} 件を解析します
                </Show>
              </p>


            </section>
          </div>
          <div class="actions setup-actions">
            <button
              type="button"
              class="primary"
              disabled={!canAnalyze()}
              onClick={() => void startAnalysis()}
            >
              {analyzeState() === "running" ? "判定中…" : "診断をはじめる"}
            </button>
          </div>
        </Show>
      </section>

      <Show when={analyzeState() === "running"}>
        <DiagnosisProgress
          progress={currentAnalysisProgress()}
          item={liveItem() ?? undefined}
          tweets={selected()}
          accountUsername={archive()?.account.username}
          accountDisplayName={archive()?.account.displayName}
          accountAvatarUrl={localAvatarUrl() ?? safeImageSource(archive()?.account.avatarUrl)}
          loadMedia={(filename) => archiveTask?.readMedia(filename) ?? Promise.resolve(null)}
          onCancel={cancelAnalysis}
        />
      </Show>

      <Show when={backend() === "wasm" && backendNote()}>
        <p class="backend-warning" role="status">{backendNote()}</p>
      </Show>

      <Show when={analysisWarning()}>
        <p class="analysis-warning" role="status">{analysisWarning()}</p>
      </Show>

      <Show when={result()}>
        {(value) => (
          <ResultCard
            result={value()}
            avatarUrl={localAvatarUrl() ?? safeImageSource(archive()?.account.avatarUrl)}
            accountUsername={archive()?.account.username}
            marginThreshold={marginThreshold()}
            onShare={() => shareResult(value())}
            onDownloadImage={() => void downloadResultImage(value())}
            shareNotice={shareNotice()}
          />
        )}
      </Show>

      <Show when={result() && analyzedItems().length > 0}>
        <TweetResults
          items={analyzedItems()}
          tweets={selected()}
          accountUsername={archive()?.account.username}
          accountDisplayName={archive()?.account.displayName}
          accountAvatarUrl={localAvatarUrl() ?? safeImageSource(archive()?.account.avatarUrl)}
        />
      </Show>

      <details class="model-details">
        <summary class="model-details__summary">実行環境とモデルの詳細</summary>
        <div class="model-details__body">
          <p>
            <a href="https://github.com/NandhaKishorM/laya" target="_blank" rel="noreferrer">
              Laya
            </a>
            のJev型判定モデルと
            <a href="https://onnxruntime.ai/docs/tutorials/web/" target="_blank" rel="noreferrer">
              ONNX Runtime Web
            </a>
            を使っています。入力から集計まで、ブラウザ内で完結する構成です。
          </p>
          <ul class="model-details__list">
            <li>各投稿を4つの二択質問として評価し、軸ごとの回答をまとめてMBTIを推定します。</li>
            <li>encoderはWebGPUで実行し、初回だけモデルを取得します。</li>
            <li>WebGPUが使えない環境では、CPU（WASM）で自動的にリトライするため、処理が大幅に遅くなります。</li>
            <li>モデルとランタイムは端末のCache Storageに保存し、投稿本文はサーバーへ送信しません。</li>
          </ul>
        </div>
      </details>
      </main>

      <footer class="footer">
        <p>
          参考:{" "}
          <a href="https://github.com/NandhaKishorM/laya" target="_blank" rel="noreferrer">
            NandhaKishorM/laya
          </a>{" "}
          /{" "}
          <a href="https://x-checkpoint.activetk.jp" target="_blank" rel="noreferrer">
            X-CheckPoint
          </a>
          。本アプリは非公式であり、X Corp. とは関係ありません。 イラスト:{" "}
          <a
            href="https://loosedrawing.com/assets/media/illustrations/png/1952.png"
            target="_blank"
            rel="noreferrer"
          >
            Loose Drawing
          </a>
        </p>
        <p class="footer__disclaimer">
          この結果は、投稿文から読み取れる言語的な傾向を統計的に集計した推定であり、医学的・心理学的な診断ではありません。
        </p>
        <p class="footer__links">
          <a href={REPO_URL} target="_blank" rel="noreferrer">
            GitHub リポジトリ
          </a>
          <a href={OWNER_GITHUB_URL} target="_blank" rel="noreferrer">
            GitHub @shibadogcap
          </a>
          <a href={OWNER_X_URL} target="_blank" rel="noreferrer">
            X @4ba_ba_baba
          </a>
        </p>
      </footer>
    </div>
  );
}
