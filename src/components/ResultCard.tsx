import { createMemo, For, Show } from "solid-js";
import type { AxisResult, MbtiResult } from "../lib/types.js";
import { formatPercent, typeAgreement } from "../lib/agreement.js";
import { safeImageSource } from "../lib/format.js";

type ResultRole = "analyst" | "diplomat" | "sentinel" | "explorer";

const RESULT_ROLES: Record<string, ResultRole> = {
  INTJ: "analyst",
  INTP: "analyst",
  ENTJ: "analyst",
  ENTP: "analyst",
  INFJ: "diplomat",
  INFP: "diplomat",
  ENFJ: "diplomat",
  ENFP: "diplomat",
  ISTJ: "sentinel",
  ISFJ: "sentinel",
  ESTJ: "sentinel",
  ESFJ: "sentinel",
  ISTP: "explorer",
  ISFP: "explorer",
  ESTP: "explorer",
  ESFP: "explorer",
};

const AXIS_ICON_PATHS: Record<AxisResult["id"], string> = {
  EI: "M4 12h16M4 12l4-4M4 12l4 4M20 12l-4-4M20 12l-4 4",
  SN: "M2.75 12S6 7 12 7s9.25 5 9.25 5-3.25 5-9.25 5-9.25-5-9.25-5ZM12 9.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5Z",
  TF: "M12 4v16M5 6h14M6 6l-3 6h6L6 6Zm12 0-3 6h6l-3-6ZM8 20h8",
  JP: "M12 4v16M5 8l7 4 7-4M5 16l7-4 7 4M12 4l-2 2M12 4l2 2",
};

function resultRole(type: string): ResultRole {
  return RESULT_ROLES[type] ?? "analyst";
}

function XIcon(props: { class?: string }) {
  return (
    <svg
      class={props.class}
      width="1em"
      height="1em"
      viewBox="0 0 24 24"
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
    </svg>
  );
}

function DownloadIcon(props: { class?: string }) {
  return (
    <svg
      class={props.class}
      width="1em"
      height="1em"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="1.9"
      stroke-linecap="round"
      stroke-linejoin="round"
      aria-hidden="true"
    >
      <path d="M12 3v12m0 0 4-4m-4 4-4-4M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
    </svg>
  );
}

function AxisIcon(props: { id: AxisResult["id"] }) {
  return (
    <svg
      class={`axis-card__icon axis-card__icon--${props.id.toLowerCase()}`}
      width="1em"
      height="1em"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-linecap="round"
      stroke-linejoin="round"
      stroke-width="1.8"
      aria-hidden="true"
    >
      <path d={AXIS_ICON_PATHS[props.id]} />
    </svg>
  );
}

function AxisCard(props: { axis: AxisResult; marginThreshold: number }) {
  const axis = props.axis;
  const positiveWidth = () => `${Math.round(axis.positiveProbability * 100)}%`;
  const chosenPercent = () => Math.round(axis.probability * 100);
  const signed = () => axis.signedBuckets ?? [0, 0, 0, 0, 0];
  const maxSigned = () => Math.max(1, ...signed());
  const signedLabel = (index: number) =>
    index === 0
      ? `${axis.negative.code}寄り`
      : index === 4
        ? `${axis.positive.code}寄り`
        : "";
  return (
    <article class={`axis-card axis-card--${axis.id.toLowerCase()}`}>
      <header class="axis-card__header">
        <div class="axis-card__title">
          <span class="axis-card__mark" aria-hidden="true">
            <AxisIcon id={axis.id} />
          </span>
          <div class="axis-card__copy">
            <span class="axis-card__name">{axis.name}</span>
            <strong class="axis-card__chosen">{axis.chosen.label}</strong>
          </div>
        </div>
        <div class="axis-card__metrics">
          <span class="axis-card__probability">
            <strong class="axis-card__percent">{chosenPercent()}%</strong>
          </span>
        </div>
      </header>
      <p class="axis-card__description axis-card__chosen-description">{axis.chosen.description}</p>
      <div class="axis-card__axis-line">
        <span class="axis-card__label axis-card__label--positive">{axis.positive.code}</span>
        <div
          class="axis-card__track"
          role="img"
          aria-label={`${axis.name}の判定は${axis.chosen.label}、${chosenPercent()}%`}
        >
          <div class="axis-card__fill" style={{ width: positiveWidth() }} />
          <div class="axis-card__mid" />
        </div>
        <span class="axis-card__label axis-card__label--negative">{axis.negative.code}</span>
      </div>
      <div
        class="axis-card__spread"
        role="img"
        aria-label={`${axis.positive.code}から${axis.negative.code}への分布。${signed()
          .map((count, index) => `${signedLabel(index)}${count}件`)
          .join("、")}`}
      >
        <span class="axis-card__spread-end">{axis.positive.code}</span>
        <div class="axis-card__spread-plot">
          {/* Reversed so the left end is the positive pole, the same side the
              track above puts it on. */}
          <For each={signed().map((count, dataIndex) => ({ count, dataIndex })).reverse()}>
            {(column) => (
              <div
                class="axis-card__spread-column"
                classList={{
                  "axis-card__spread-column--chosen":
                    (column.dataIndex >= 2) === (axis.positiveProbability >= 0.5),
                }}
                title={`${signedLabel(column.dataIndex)}: ${column.count}件`}
              >
                <span class="axis-card__spread-count">{column.count}</span>
                <span
                  class="axis-card__spread-bar"
                  style={{ height: `${Math.max(4, (column.count / maxSigned()) * 100)}%` }}
                />
              </div>
            )}
          </For>
          <span class="axis-card__spread-mid" aria-hidden="true" />
        </div>
        <span class="axis-card__spread-end">{axis.negative.code}</span>
      </div>
    </article>
  );
}

export function ResultCard(props: {
  result: MbtiResult;
  avatarUrl?: string;
  accountUsername?: string;
  marginThreshold?: number;
  live?: boolean;
  onShare?: () => void;
  onDownloadImage?: () => void;
  shareNotice?: string | null;
}) {
  /**
   * The headline number and every candidate row use this one measure: the mean
   * of the four axis probabilities of that type. The posterior product stays as
   * the sort key, because it is what makes the ordering a distribution, but
   * showing it next to a mean-based headline read as two different numbers.
   */
  const topAgreement = () => typeAgreement(props.result.ranking[0]);
  const alternatives = createMemo(() =>
    props.result.ranking
      .slice(1)
      .map((entry) => ({ entry, agreement: typeAgreement(entry) }))
      .sort((a, b) => b.agreement - a.agreement || a.entry.rank - b.entry.rank),
  );

  return (
    <section
      class={`result result--${resultRole(props.result.type)}`}
      classList={{
        "result--live": props.live,
        "result--final": !props.live,
      }}
      aria-label={props.live ? "途中の診断結果" : "最終診断結果"}
    >
      <div class="result__hero">
        <div class="result__hero-main">
          <Show
            when={safeImageSource(props.avatarUrl)}
            fallback={
              <span class="result__avatar result__avatar--fallback" aria-hidden="true">
                <span class="result__initial">
                  {(props.accountUsername?.slice(0, 1) ?? "X").toUpperCase()}
                </span>
              </span>
            }
          >
            {(url) => <img class="result__avatar" src={url()} alt="" />}
          </Show>
          <div class="result__identity">
            <Show when={props.accountUsername}>
              <span class="result__account">@{props.accountUsername}</span>
            </Show>
            <div class="result__type-word" role="img" aria-label={props.result.type}>
              <span class="result__type-letters" aria-hidden="true">
                {props.result.type}
              </span>
            </div>
          </div>
        </div>
        <div class="result__confidence">
          <strong class="result__confidence-value" aria-label={`4軸平均一致度 ${formatPercent(topAgreement())}`}>
            {formatPercent(topAgreement())}
          </strong>
        </div>
      </div>

      <div class="result__axes">
        <For each={props.result.axes}>
          {(axis) => <AxisCard axis={axis} marginThreshold={props.marginThreshold ?? 0.1} />}
        </For>
      </div>

      <Show when={!props.live && props.result.ranking.length >= 3}>
        <section class="result__rankings" aria-label="その他のMBTI候補">
          <div class="result__rankings-heading">
            <span>その他の候補</span>
            <small>同じ4軸平均一致度の順</small>
          </div>
          <div class="result__rankings-list">
            <For each={alternatives()}>
              {(row, index) => (
                <div
                  class="result__ranking"
                  classList={{ "result__ranking--farthest": row.entry.flips >= 3 }}
                  title={`上位から${row.entry.flips}軸異なる候補`}
                >
                  <span class="result__ranking-rank">{index() + 2}位</span>
                  <strong>{row.entry.type}</strong>
                  <span class="result__ranking-percent">{formatPercent(row.agreement)}</span>
                </div>
              )}
            </For>
          </div>
        </section>
      </Show>

      <Show when={!props.live && (props.onShare || props.onDownloadImage)}>
        <section class="result__share" aria-label="結果を共有">
          <p class="result__share-heading">この結果を共有する</p>
          <div class="result__share-actions">
            <Show when={props.onShare}>
              <button
                type="button"
                class="share-button share-button--x"
                onClick={() => props.onShare?.()}
                aria-label="Xで診断結果を共有"
              >
                <span class="share-button__mark" aria-hidden="true">
                  <XIcon class="share-button__icon" />
                </span>
                <span>Xで共有</span>
              </button>
            </Show>
            <Show when={props.onDownloadImage}>
              <button
                type="button"
                class="share-button share-button--image"
                onClick={() => props.onDownloadImage?.()}
                aria-label="診断結果を画像でダウンロード"
              >
                <span class="share-button__mark" aria-hidden="true">
                  <DownloadIcon class="share-button__icon" />
                </span>
                <span>画像を保存</span>
              </button>
            </Show>
          </div>
          <Show when={props.shareNotice}>
            <span class="share-notice" role="status">{props.shareNotice}</span>
          </Show>
        </section>
      </Show>
    </section>
  );
}
