import { createEffect, createSignal, For, onCleanup, Show } from "solid-js";
import { AXES } from "../lib/mbti.js";
import { formatDuration, safeImageSource } from "../lib/format.js";
import {
  chosenPole,
  itemType,
  shortText,
  TWEET_COUNT,
  TWEET_TIME,
  type PoleReading,
} from "../lib/reading.js";
import type { AnalyzedItem, AnalyzeProgress } from "../lib/inference.js";
import type { Tweet, TweetMedia } from "../lib/types.js";

export interface DiagnosisProgressProps {
  progress: AnalyzeProgress;
  item?: AnalyzedItem;
  tweets: Tweet[];
  accountUsername?: string;
  accountDisplayName?: string;
  accountAvatarUrl?: string;
  loadMedia?: (filename: string) => Promise<Blob | null>;
  onCancel?: () => void;
}

type TweetMetricKey = "favoriteCount" | "retweetCount" | "replyCount";

interface TweetMetric {
  key: TweetMetricKey;
  label: string;
  path: string;
}

const TIPS = [
  "各投稿を4つの軸に分けて、傾向を集計しています。",
  "回答が近い投稿は弱い根拠として扱い、結果を安定させます。",
  "同じ軸でも、確信度が高いほど大きく反映します。",
  "判定はブラウザ内で行い、投稿本文を送信しません。",
];

const TWEET_METRICS: readonly TweetMetric[] = [
  {
    key: "replyCount",
    label: "返信",
    path: "M21 11.5a8.4 8.4 0 0 1-9 8.5 9.5 9.5 0 0 1-4-.9L3 21l1.9-4.5A8.5 8.5 0 1 1 21 11.5Z",
  },
  {
    key: "retweetCount",
    label: "リポスト",
    path: "m17 2 4 4-4 4M3 11V9a3 3 0 0 1 3-3h15M7 22l-4-4 4-4m14-1v2a3 3 0 0 1-3 3H3",
  },
  {
    key: "favoriteCount",
    label: "いいね",
    path: "M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.7l-1.1-1.1a5.5 5.5 0 0 0-7.8 7.8l1.1 1.1L12 21l7.8-7.5 1.1-1.1a5.5 5.5 0 0 0-.1-7.8Z",
  },
];

function XMark(props: { size?: number; class?: string }) {
  return (
    <svg
      class={props.class}
      viewBox="0 0 24 24"
      width={props.size ?? 16}
      height={props.size ?? 16}
      aria-hidden="true"
    >
      <path
        d="M18.9 2H22l-7.1 8.1L23.2 22h-6.6l-5.2-6.8L5.5 22H2.3l7.6-8.7L1.9 2h6.8l4.7 6.2L18.9 2Zm-1.2 18h1.7L7.4 3.8H5.6L17.7 20Z"
        fill="currentColor"
      />
    </svg>
  );
}

function MetricIcon(props: { path: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      width="14"
      height="14"
      fill="none"
      stroke="currentColor"
      stroke-width="1.8"
      stroke-linecap="round"
      stroke-linejoin="round"
      aria-hidden="true"
    >
      <path d={props.path} />
    </svg>
  );
}

function PostHeader(props: {
  tweet?: Tweet;
  accountUsername?: string;
  accountDisplayName?: string;
  accountAvatarUrl?: string;
}) {
  const username = () => props.tweet?.username?.trim() || props.accountUsername?.trim() || null;
  const displayName = () =>
    props.tweet?.displayName?.trim() || props.accountDisplayName?.trim() || null;
  const avatarUrl = () =>
    safeImageSource(props.tweet?.avatarUrl) ?? safeImageSource(props.accountAvatarUrl);
  const createdAt = () => {
    const value = props.tweet?.createdAt;
    if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
      return null;
    }
    return Number.isNaN(new Date(value).getTime()) ? null : value;
  };

  return (
    <div class="reading__header">
      <span class="reading__avatar" aria-hidden="true">
        <Show
          when={avatarUrl()}
          fallback={
            <span class="reading__initial">
              {(username()?.slice(0, 1) ?? "X").toUpperCase()}
            </span>
          }
        >
          {(url) => <img class="reading__avatar-image" src={url()} alt="" />}
        </Show>
      </span>
      <div class="reading__identity">
        <div class="reading__byline">
          <Show when={displayName()}>
            {(value) => <span class="reading__display-name">{value()}</span>}
          </Show>
          <Show
            when={username()}
            fallback={<b class="reading__username">アーカイブの投稿</b>}
          >
            {(value) => (
              <b class="reading__username" title={value()}>
                @{value()}
              </b>
            )}
          </Show>
          <Show when={createdAt()}>
            {(value) => (
              <span class="reading__time">
                {" · "}
                <time dateTime={new Date(value()).toISOString()}>
                  {TWEET_TIME.format(value())}
                </time>
              </span>
            )}
          </Show>
        </div>
      </div>
      <span class="reading__source" aria-hidden="true">
        <XMark size={15} class="reading__source-mark" />
      </span>
    </div>
  );
}

function isTweetCount(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

function formatTweetCount(value: unknown): string {
  return isTweetCount(value) ? TWEET_COUNT.format(value) : "";
}

export function DiagnosisProgress(props: DiagnosisProgressProps) {
  const percent = () =>
    props.progress.total > 0
      ? Math.min(100, Math.max(0, Math.round((props.progress.done / props.progress.total) * 100)))
      : 0;
  const [tipIndex, setTipIndex] = createSignal(0);
  const tipTimer = window.setInterval(() => {
    setTipIndex((index) => (index + 1) % TIPS.length);
  }, 7000);
  const tip = () => TIPS[tipIndex()];
  const latest = () => props.item;
  const tweetsById = new Map(props.tweets.map((tweet) => [tweet.id, tweet]));
  const currentMedia = (): TweetMedia[] => {
    const item = latest();
    return item ? (tweetsById.get(item.id)?.media ?? []) : [];
  };
  const photoMedia = () => currentMedia().filter((media) => media.type === "photo").slice(0, 4);
  const videoMedia = () => currentMedia().find((media) => media.type !== "photo");
  const [mediaUrls, setMediaUrls] = createSignal<Record<string, string>>({});
  let mediaObjectUrls: string[] = [];
  let mediaRequest = 0;

  createEffect(() => {
    const media = photoMedia();
    const request = ++mediaRequest;
    for (const url of mediaObjectUrls) URL.revokeObjectURL(url);
    mediaObjectUrls = [];
    if (media.length === 0 || !props.loadMedia) {
      setMediaUrls({});
      return;
    }
    void Promise.all(
      media.map(async (item) => [item.filename, await props.loadMedia!(item.filename)] as const),
    ).then((entries) => {
      if (request !== mediaRequest) return;
      const urls: Record<string, string> = {};
      for (const [filename, blob] of entries) {
        if (!blob) continue;
        const url = URL.createObjectURL(blob);
        mediaObjectUrls.push(url);
        urls[filename] = url;
      }
      setMediaUrls(urls);
    });
  });

  onCleanup(() => {
    window.clearInterval(tipTimer);
    for (const url of mediaObjectUrls) URL.revokeObjectURL(url);
  });

  return (
    <section
      class="diagnosis diagnosis--progress"
      aria-live="polite"
      aria-busy="true"
    >
      <div class="diagnosis__header">
        <div class="diagnosis__title">
          <h2>投稿を読み取っています</h2>
        </div>
        <div class="diagnosis__summary">
          <div class="diagnosis__count">
            <strong>{percent()}%</strong>
            <span>
              {props.progress.done.toLocaleString()} / {props.progress.total.toLocaleString()}
            </span>
          </div>
          <Show when={props.onCancel}>
            <button
              type="button"
              class="icon-button diagnosis__cancel"
              aria-label="診断を中断"
              title="診断を中断"
              onClick={() => props.onCancel?.()}
            >
              <span class="diagnosis__cancel-mark" aria-hidden="true">
                ×
              </span>
            </button>
          </Show>
        </div>
      </div>

      <div
        class="progress-track diagnosis__progress"
        role="progressbar"
        aria-valuemin="0"
        aria-valuemax="100"
        aria-valuenow={percent()}
        aria-valuetext={`${percent()}%、${props.progress.done} / ${props.progress.total}`}
        aria-label="診断の進捗"
      >
        <div
          class="progress-track__fill"
          style={{ width: `${percent()}%` }}
        />
      </div>
      <div
        class="diagnosis__meta diagnosis__timing"
        style={{ color: "var(--ink-soft)", "font-size": "12px" }}
      >
        <span class="diagnosis__elapsed">
          経過 {formatDuration(props.progress.elapsedMs)}
        </span>
        <span class="diagnosis__eta">
          残り目安 {formatDuration(props.progress.etaMs)}
        </span>
      </div>

      <Show
        when={latest()}
        fallback={
          <article class="reading diagnosis__reading">
            <PostHeader
              accountUsername={props.accountUsername}
              accountDisplayName={props.accountDisplayName}
              accountAvatarUrl={props.accountAvatarUrl}
            />
            <p class="reading__placeholder">最初の投稿を準備しています。</p>
          </article>
        }
      >
        {(item) => {
          const tweet = () => tweetsById.get(item().id);
          const text = () => shortText(tweet()?.text ?? "投稿文を表示できません");
          const hasMetrics = () =>
            TWEET_METRICS.some((metric) => isTweetCount(tweet()?.[metric.key]));

          return (
            <article class="reading diagnosis__reading">
              <PostHeader
                tweet={tweet()}
                accountUsername={props.accountUsername}
                accountDisplayName={props.accountDisplayName}
                accountAvatarUrl={props.accountAvatarUrl}
              />
              <p class="reading__text">{text()}</p>
              <Show when={photoMedia().length > 0}>
                <div
                  class="reading__media-grid"
                  classList={{ "reading__media-grid--single": photoMedia().length === 1 }}
                >
                  <For each={photoMedia()}>
                    {(media) => (
                      <Show when={mediaUrls()[media.filename]}>
                        {(url) => (
                          <img
                            class="reading__media"
                            src={url()}
                            alt=""
                            loading="lazy"
                            decoding="async"
                            style={
                              media.width && media.height
                                ? `aspect-ratio: ${media.width} / ${media.height}`
                                : undefined
                            }
                          />
                        )}
                      </Show>
                    )}
                  </For>
                </div>
              </Show>
              <Show when={videoMedia()}>
                {(video) => (
                  <span class="reading__media-note">
                    動画 {video().durationMs ? `${Math.round(video().durationMs! / 1000)}秒` : ""}
                  </span>
                )}
              </Show>

              <div class="reading__poles" role="list" aria-label="この投稿の4つの傾向">
                <span class="reading__pole reading__pole--mbti" role="listitem">
                  <b>#{itemType(item())}</b>
                </span>
                <For each={AXES}>
                  {(axis) => {
                    const reading = (): PoleReading | null => chosenPole(axis, item());
                    return (
                      <span
                        class={`reading__pole reading__pole--${axis.id.toLowerCase()}`}
                        role="listitem"
                        aria-label={`${reading()?.label ?? "判定待ち"} ${reading()?.share ?? 0}%`}
                      >
                        <b>{reading()?.label ?? "#判定待ち"}</b>
                      </span>
                    );
                  }}
                </For>
              </div>

              <Show when={hasMetrics()}>
                <div class="reading__metrics" aria-label="投稿の反応">
                  <For each={TWEET_METRICS}>
                    {(metric) => {
                      const count = () => tweet()?.[metric.key];
                      return (
                        <Show when={isTweetCount(count())}>
                          <span
                            class={`reading__metric reading__metric--${metric.key}`}
                            aria-label={`${metric.label} ${formatTweetCount(count())}`}
                            title={metric.label}
                          >
                            <MetricIcon path={metric.path} />
                            <span class="reading__metric-value">
                              {formatTweetCount(count())}
                            </span>
                          </span>
                        </Show>
                      );
                    }}
                  </For>
                </div>
              </Show>
            </article>
          );
        }}
      </Show>

      <div class="tip-box diagnosis__tip">
        <span class="tip-box__icon" aria-hidden="true">
          ↗
        </span>
        <div class="tip-box__body">
          <span class="tip-box__label">ヒント</span>
          <p class="tip-box__text">{tip()}</p>
        </div>
      </div>
    </section>
  );
}
