import { createMemo, createSignal, For, Show } from "solid-js";
import { itemPoles, itemType, shortText, TWEET_TIME } from "../lib/reading.js";
import { safeImageSource } from "../lib/format.js";
import type { AnalyzedItem } from "../lib/inference.js";
import type { Tweet } from "../lib/types.js";

export interface TweetResultsProps {
  items: AnalyzedItem[];
  tweets: Tweet[];
  accountUsername?: string;
  accountDisplayName?: string;
  accountAvatarUrl?: string;
}

/** Opened with a short read; the rest arrives in bigger steps. */
const FIRST_PAGE = 5;
const PAGE_SIZE = 20;

/**
 * The per-post breakdown of a finished run, laid out like a timeline: the
 * aggregate card only shows totals, and this is where a reader can see which
 * posts carried each pole. Long runs are paged instead of rendering every post
 * at once, so opening the section stays instant.
 */
export function TweetResults(props: TweetResultsProps) {
  const [visible, setVisible] = createSignal(FIRST_PAGE);
  const tweetsById = createMemo(() => new Map(props.tweets.map((tweet) => [tweet.id, tweet])));
  const ordered = createMemo(() =>
    [...props.items].sort((a, b) => {
      const left = tweetsById().get(a.id)?.createdAt ?? 0;
      const right = tweetsById().get(b.id)?.createdAt ?? 0;
      return left - right || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);
    }),
  );
  const rows = createMemo(() =>
    ordered()
      .map((item) => ({ item, tweet: tweetsById().get(item.id) }))
      .filter((row) => row.tweet !== undefined),
  );
  const shown = createMemo(() => rows().slice(0, visible()));
  const remaining = () => rows().length - shown().length;
  const avatar = () => safeImageSource(props.accountAvatarUrl);

  return (
    <details class="tweet-results" open>
      <summary class="tweet-results__summary">
        <span class="tweet-results__summary-label">投稿ごとの解析結果</span>
        <span class="tweet-results__summary-total">
          {rows().length.toLocaleString()} / {props.items.length.toLocaleString()}件
        </span>
      </summary>
      <div class="tweet-results__body">
        <ol class="timeline">
          <For each={shown()}>
            {(row) => (
              <li class="timeline__post">
                <div class="timeline__avatar">
                  <Show
                    when={avatar()}
                    fallback={
                      <span class="timeline__avatar-fallback" aria-hidden="true">
                        {(props.accountUsername?.slice(0, 1) ?? "X").toUpperCase()}
                      </span>
                    }
                  >
                    {(url) => <img src={url()} alt="" loading="lazy" decoding="async" />}
                  </Show>
                </div>
                <div class="timeline__main">
                  <p class="timeline__head">
                    <strong class="timeline__name">
                      {props.accountDisplayName ?? (props.accountUsername ? `@${props.accountUsername}` : "このアカウント")}
                    </strong>
                    <span class="timeline__meta">
                      <Show when={props.accountUsername}>
                        <span class="timeline__handle">@{props.accountUsername}</span>
                      </Show>
                      <time
                        class="timeline__time"
                        datetime={new Date(row.tweet!.createdAt).toISOString()}
                      >
                        {TWEET_TIME.format(row.tweet!.createdAt)}
                      </time>
                    </span>
                  </p>
                  <p class="timeline__text">{shortText(row.tweet!.text)}</p>
                  <div class="timeline__poles">
                    <span class="timeline__type">#{itemType(row.item)}</span>
                    <For each={itemPoles(row.item)}>
                      {(entry) => (
                        <span
                          class={`timeline__pole timeline__pole--${entry.axis.id.toLowerCase()}`}
                          title={`${entry.axis.name}: ${entry.pole.share}%`}
                        >
                          {entry.pole.label}
                          <em>{entry.pole.share}%</em>
                        </span>
                      )}
                    </For>
                  </div>
                </div>
              </li>
            )}
          </For>
        </ol>
        <Show when={remaining() > 0}>
          <button
            type="button"
            class="tweet-results__more"
            onClick={() => setVisible((count) => count + PAGE_SIZE)}
          >
            さらに {PAGE_SIZE} 件表示（残り {remaining().toLocaleString()} 件）
          </button>
        </Show>
        <Show when={rows().length === 0}>
          <p class="tweet-results__empty">解析済みの投稿がありません。</p>
        </Show>
      </div>
    </details>
  );
}
