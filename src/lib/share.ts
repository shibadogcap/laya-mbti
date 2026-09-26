import type { MbtiResult } from "./types.js";

export interface ResultImageInput {
  result: MbtiResult;
  /** Displayed next to the type; also used for the avatar fallback initial. */
  accountUsername?: string;
  /** Same-origin URL or blob: URL of the archived profile image. */
  avatarUrl?: string;
  /** 0..1 average of the four axis probabilities, the same number the card shows. */
  agreement: number;
  /** Display strings for the alternative types, already formatted by the caller. */
  alternatives: { type: string; percent: string }[];
}

const WIDTH = 1080;
const PAD = 60;
const LEFT = 120;
const RIGHT = WIDTH - 120;
const CONTENT = RIGHT - LEFT;
/** Alternatives per row: five left no room for a four-letter type and a score. */
const ALT_COLUMNS = 4;
const ALT_ROWS = 4;
const FONT = '"Noto Sans JP Variable", "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif';

const INK = "#18212b";
const SOFT = "#5b6672";
const FAINT = "#8b95a1";
const LINE = "#dfe3e8";
const CARD = "#ffffff";
const CANVAS = "#f4f5f7";

const AXIS_COLORS = ["#5b4eaa", "#397e65", "#a36b4c", "#8a6a1f"];

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
): void {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + width, y, x + width, y + height, radius);
  ctx.arcTo(x + width, y + height, x, y + height, radius);
  ctx.arcTo(x, y + height, x, y, radius);
  ctx.arcTo(x, y, x + width, y, radius);
  ctx.closePath();
}

async function loadAvatar(url: string | undefined): Promise<HTMLImageElement | null> {
  if (!url) return null;
  try {
    const image = new Image();
    image.decoding = "async";
    image.src = url;
    await image.decode();
    return image;
  } catch {
    return null;
  }
}

function drawCover(
  ctx: CanvasRenderingContext2D,
  image: HTMLImageElement,
  x: number,
  y: number,
  size: number,
): void {
  const scale = Math.max(size / image.width, size / image.height);
  const width = image.width * scale;
  const height = image.height * scale;
  ctx.save();
  roundRect(ctx, x, y, size, size, size / 2);
  ctx.clip();
  ctx.drawImage(image, x + (size - width) / 2, y + (size - height) / 2, width, height);
  ctx.restore();
}

/** Draws text with an explicit per-character advance, so tracking is even. */
function drawTracked(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  tracking: number,
): number {
  let cursor = x;
  for (const char of text) {
    ctx.fillText(char, cursor, y);
    cursor += ctx.measureText(char).width + tracking;
  }
  return cursor - x - tracking;
}

function trackedWidth(ctx: CanvasRenderingContext2D, text: string, tracking: number): number {
  let total = 0;
  for (const char of text) total += ctx.measureText(char).width + tracking;
  return total - tracking;
}

function rule(ctx: CanvasRenderingContext2D, x: number, y: number, right: number): void {
  ctx.strokeStyle = LINE;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(right, y);
  ctx.stroke();
}

function ellipsize(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string {
  if (ctx.measureText(text).width <= maxWidth) return text;
  let cut = text;
  while (cut.length > 1 && ctx.measureText(`${cut}…`).width > maxWidth) {
    cut = cut.slice(0, -1);
  }
  return `${cut}…`;
}

/**
 * Draws one line inside `[x, x + maxWidth]`, shrinking the font before it
 * ellipsizes. Canvas has no text overflow, so a long handle or a long
 * disclaimer used to run straight past the card edge.
 */
function drawFitted(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  maxWidth: number,
  options: { weight?: number; size: number; minSize?: number; align?: "left" | "right" },
): void {
  const weight = options.weight ?? 700;
  const minSize = options.minSize ?? Math.max(11, Math.round(options.size * 0.62));
  let size = options.size;
  for (; size > minSize; size -= 1) {
    ctx.font = `${weight} ${size}px ${FONT}`;
    if (ctx.measureText(text).width <= maxWidth) break;
  }
  const fitted = ellipsize(ctx, text, maxWidth);
  const width = ctx.measureText(fitted).width;
  const drawX = options.align === "right" ? x - width : x;
  ctx.fillText(fitted, drawX, y);
}

/**
 * Draws the shareable result card and returns it as a PNG blob.
 *
 * The layout is a strict top-down stack: every block reserves its own band and
 * nothing is placed by eye, because canvas has no line box to reflow into and
 * an earlier version drew the type straight through the avatar.
 */
export async function renderResultImage(input: ResultImageInput): Promise<Blob> {
  if (typeof document === "undefined") {
    throw new Error("画像を生成できる環境ではありません。");
  }
  await document.fonts?.ready;
  const alternatives = input.alternatives.slice(0, ALT_COLUMNS * ALT_ROWS);

  // The height follows the content, so the card never ends with a block of
  // empty canvas and no band can be pushed off the bottom.
  const HEIGHT =
    PAD +
    64 + // header band: avatar, handle and type side by side
    76 + // gap
    46 + // gap
    2 + // rule
    52 + // gap
    input.result.axes.length * 76 + // axis rows
    44 + // gap
    2 + // rule
    50 + // gap
    44 + // "other candidates"
    Math.max(1, Math.ceil(alternatives.length / ALT_COLUMNS)) * 48 + // rows
    40 + // gap
    26 + // disclaimer
    PAD;

  const canvas = document.createElement("canvas");
  canvas.width = WIDTH;
  canvas.height = HEIGHT;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("画像生成に対応していないブラウザです。");

  ctx.fillStyle = CANVAS;
  ctx.fillRect(0, 0, WIDTH, HEIGHT);
  ctx.fillStyle = CARD;
  roundRect(ctx, PAD, PAD, WIDTH - PAD * 2, HEIGHT - PAD * 2, 36);
  ctx.fill();

  const left = LEFT;
  const right = RIGHT;
  const content = CONTENT;
  let y = PAD + 56;

  // 1. Header, laid out like the result card: avatar on the left, handle above
  // the type beside it, score on the right sharing the type's baseline.
  const AVATAR = 112;
  const HANDLE_SIZE = 30;
  const TYPE_SIZE = 132;
  const TRACKING = -7;
  const typeBaseline = y + AVATAR + 6;
  const handleBaseline = y + 26;

  const avatar = await loadAvatar(input.avatarUrl);
  if (avatar) {
    drawCover(ctx, avatar, left, y, AVATAR);
  } else {
    ctx.fillStyle = INK;
    roundRect(ctx, left, y, AVATAR, AVATAR, AVATAR / 2);
    ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.font = `800 52px ${FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(
      (input.accountUsername?.slice(0, 1) ?? "X").toUpperCase(),
      left + AVATAR / 2,
      y + AVATAR / 2,
    );
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
  }

  const identityX = left + AVATAR + 24;
  if (input.accountUsername) {
    ctx.fillStyle = SOFT;
    drawFitted(ctx, `@${input.accountUsername}`, identityX, handleBaseline, content - AVATAR - 24 - 200, {
      size: HANDLE_SIZE,
      weight: 700,
    });
  }

  ctx.font = `850 ${TYPE_SIZE}px ${FONT}`;
  const typeWidth = trackedWidth(ctx, input.result.type, TRACKING);
  const percentText = `${Math.round(input.agreement * 100)}%`;
  ctx.font = `850 84px ${FONT}`;
  const percentWidth = ctx.measureText(percentText).width;
  const inline = identityX + typeWidth + 48 + percentWidth <= right;
  ctx.fillStyle = INK;
  drawTracked(ctx, input.result.type, identityX, typeBaseline, TRACKING);
  if (inline) {
    ctx.fillText(percentText, right - percentWidth, typeBaseline);
  } else {
    drawFitted(ctx, percentText, identityX, typeBaseline + 82, content - AVATAR - 24, {
      size: 52,
      weight: 850,
    });
    y = typeBaseline + 82;
  }
  if (inline) y = typeBaseline;

  y += 46;
  rule(ctx, left, y, right);
  y += 52;

  // 3. One band per axis: letters, chosen pole, bar, share.
  input.result.axes.forEach((axis, index) => {
    const color = AXIS_COLORS[index % AXIS_COLORS.length];
    const chosen = Math.round(axis.probability * 100);
    ctx.fillStyle = SOFT;
    drawFitted(ctx, `${axis.positive.code} / ${axis.negative.code}`, left, y, 150, { size: 28, weight: 700 });
    ctx.fillStyle = color;
    drawFitted(ctx, axis.chosen.label, left + 160, y, content - 160 - 130, { size: 32, weight: 850 });
    drawFitted(ctx, `${chosen}%`, right, y, 120, { size: 32, weight: 850, align: "right" });

    const barTop = y + 18;
    ctx.fillStyle = "#eceff2";
    roundRect(ctx, left, barTop, content, 14, 7);
    ctx.fill();
    ctx.fillStyle = color;
    roundRect(ctx, left, barTop, Math.max(14, (content * chosen) / 100), 14, 7);
    ctx.fill();
    y += 76;
  });

  y += 44;
  rule(ctx, left, y, right);
  y += 50;

  // 4. Alternatives, four per row so a four-letter type always fits.
  ctx.fillStyle = FAINT;
  drawFitted(ctx, "その他の候補", left, y, content, { size: 26, weight: 700 });
  y += 44;
  const columnWidth = content / ALT_COLUMNS;
  alternatives.forEach((entry, index) => {
    const column = index % ALT_COLUMNS;
    const row = Math.floor(index / ALT_COLUMNS);
    const x = left + column * columnWidth;
    const lineY = y + row * 48;
    const cell = columnWidth - 22;
    ctx.fillStyle = SOFT;
    drawFitted(ctx, `${index + 2}位`, x, lineY, 46, { size: 24, weight: 700 });
    ctx.fillStyle = INK;
    drawFitted(ctx, entry.type, x + 52, lineY, cell - 52 - 76, { size: 30, weight: 800 });
    ctx.fillStyle = FAINT;
    drawFitted(ctx, entry.percent, x + cell, lineY, 72, { size: 24, weight: 700, align: "right" });
  });

  ctx.fillStyle = FAINT;
  drawFitted(
    ctx,
    `laya-mbti · ${input.result.usedTweets.toLocaleString()}件の投稿から推定（投稿文から読み取れる傾向の集計であり、心理学的診断ではありません）`,
    left,
    HEIGHT - PAD - 28,
    content,
    { size: 22, weight: 600 },
  );

  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("画像の書き出しに失敗しました。"));
    }, "image/png");
  });
}

/** Renders the card and saves it as a PNG the browser downloads. */
export async function downloadResultImage(input: ResultImageInput): Promise<void> {
  const blob = await renderResultImage(input);
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `laya-mbti-${input.result.type}.png`;
  anchor.rel = "noopener";
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
}
