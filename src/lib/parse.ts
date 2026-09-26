import type { AccountInfo, Tweet, TweetMedia, TweetMediaType } from "./types.js";

export interface ProfileInfo {
  avatarMediaUrl?: string;
  headerMediaUrl?: string;
}

/** Matches `data/tweets.js`, `data/tweets-part1.js`, `data/tweet.js`, `data/tweets.csv`. */
export const TWEETS_RE = /(^|\/)data\/tweets?(-part\d+)?\.(js|csv)$/i;
/** Matches `data/account.js` and `data/account-part1.js`. */
export const ACCOUNT_RE = /(^|\/)data\/account(-part\d+)?\.js$/i;
/** Matches `data/profile.js` and `data/profile-part1.js`. */
export const PROFILE_RE = /(^|\/)data\/profile(-part\d+)?\.js$/i;
/** Matches the bundled current profile avatar/header image files. */
export const PROFILE_MEDIA_RE = /(^|\/)data\/profile_media\/[^/]+$/i;
/** Matches the bundled tweet media files, e.g. `data/tweets_media/1234/1234_Ab.jpg`. */
export const TWEETS_MEDIA_RE = /(^|\/)data\/tweets_media\/.+$/i;

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function optionalText(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const text = value.trim();
  return text.length > 0 ? text : undefined;
}

function optionalCount(value: unknown): number | undefined {
  if (typeof value === "number") {
    return Number.isSafeInteger(value) && value >= 0 ? value : undefined;
  }
  // Current archives serialize engagement counts as strings, e.g. "12".
  if (typeof value === "string") {
    const digits = value.trim();
    if (!/^\d+$/.test(digits)) return undefined;
    const count = Number(digits);
    return Number.isSafeInteger(count) ? count : undefined;
  }
  return undefined;
}

function optionalPositiveInt(value: unknown): number | undefined {
  if (typeof value === "number") {
    return Number.isInteger(value) && value > 0 ? value : undefined;
  }
  if (typeof value === "string" && /^\d+$/.test(value.trim())) {
    const parsed = Number(value);
    return Number.isInteger(parsed) && parsed > 0 ? parsed : undefined;
  }
  return undefined;
}

/** Last path segment of an archive-style URL or path, without query or fragment. */
export function mediaBasename(url: string): string | undefined {
  const name = url.split(/[?#]/)[0].split("/").at(-1);
  return name !== undefined && name.length > 0 ? name : undefined;
}

function parseJsonArray(text: string): unknown[] {
  const start = text.indexOf("[");
  if (start < 0) return [];
  try {
    const source = text.slice(start).trim().replace(/;$/, "");
    const raw: unknown = JSON.parse(source);
    return Array.isArray(raw) ? raw : [];
  } catch {
    return [];
  }
}

const NAMED_ENTITIES: Record<string, string> = {
  amp: "&",
  lt: "<",
  gt: ">",
  quot: '"',
  apos: "'",
  nbsp: " ",
};

const ENTITY_RE = /&(#x[0-9a-f]+|#\d+|[a-z]+);/gi;

/**
 * X stores `full_text` HTML-escaped, so an ampersand arrives as `&amp;` and an
 * angle bracket as `&lt;`. Left alone it reaches the model, the deduplication
 * fingerprint and the reading card as literal entity text, which is why `&`
 * looked wrong on screen. One pass, so `&amp;lt;` decodes to `&lt;` and not to
 * `<`.
 */
export function decodeEntities(text: string): string {
  if (!text.includes("&")) return text;
  return text.replace(ENTITY_RE, (match, body: string) => {
    if (body[0] === "#") {
      const hex = body[1] === "x" || body[1] === "X";
      const code = Number.parseInt(body.slice(hex ? 2 : 1), hex ? 16 : 10);
      if (!Number.isFinite(code) || code < 0 || code > 0x10ffff) return match;
      try {
        return String.fromCodePoint(code);
      } catch {
        return match;
      }
    }
    return NAMED_ENTITIES[body.toLowerCase()] ?? match;
  });
}

function toTweet(
  body: unknown,
  id: unknown,
  createdAt: unknown,
): Tweet | null {
  if (typeof body !== "string" || body.length === 0) return null;
  const created = Date.parse(String(createdAt ?? ""));
  return {
    id: String(id ?? ""),
    createdAt: Number.isFinite(created) ? created : 0,
    text: decodeEntities(body),
  };
}

function addJsMetadata(tweet: Tweet, source: Record<string, unknown>): Tweet {
  const user = asRecord(source.user);
  const username = optionalText(user?.screen_name);
  const displayName = optionalText(user?.name);
  const avatarUrl = optionalText(
    user?.profile_image_url_https ?? user?.profile_image_url ?? user?.avatar_image_url,
  );
  const favoriteCount = optionalCount(source.favorite_count);
  const retweetCount = optionalCount(source.retweet_count);
  const replyCount = optionalCount(source.reply_count);
  const language = optionalText(source.lang);
  const media = parseTweetMedia(tweet.id, source);
  const isRetweet =
    source.retweeted === true ||
    source.retweeted_status_id != null ||
    source.retweeted_status != null;
  const isReply = source.in_reply_to_status_id != null || source.in_reply_to_user_id != null;

  if (username !== undefined) tweet.username = username;
  if (displayName !== undefined) tweet.displayName = displayName;
  if (avatarUrl !== undefined) tweet.avatarUrl = avatarUrl;
  if (favoriteCount !== undefined) tweet.favoriteCount = favoriteCount;
  if (retweetCount !== undefined) tweet.retweetCount = retweetCount;
  if (replyCount !== undefined) tweet.replyCount = replyCount;
  if (language !== undefined) tweet.language = language;
  if (media.length > 0) tweet.media = media;
  if (isRetweet) tweet.isRetweet = true;
  if (isReply) tweet.isReply = true;

  return tweet;
}

function mediaType(record: Record<string, unknown>): TweetMediaType {
  const type = optionalText(record.type)?.toLowerCase();
  if (type === "video" || type === "animated_gif" || type === "photo") return type;
  return record.video_info ? "video" : "photo";
}

/** Highest-bitrate `video/mp4` variant, the only format the archive can serve. */
function mp4Variant(
  videoInfo: Record<string, unknown> | null,
): { url: string; contentType: string } | undefined {
  const variants = videoInfo?.variants;
  if (!Array.isArray(variants)) return undefined;
  let best: { url: string; contentType: string; bitrate: number } | undefined;
  for (const variant of variants) {
    const record = asRecord(variant);
    const contentType = optionalText(record?.content_type)?.toLowerCase();
    if (contentType !== "video/mp4") continue;
    const url = optionalText(record?.url);
    if (!url) continue;
    const bitrate =
      typeof record?.bitrate === "number" && Number.isFinite(record.bitrate) && record.bitrate >= 0
        ? record.bitrate
        : 0;
    if (!best || bitrate > best.bitrate) best = { url, contentType, bitrate };
  }
  return best ? { url: best.url, contentType: best.contentType } : undefined;
}

function mediaDimensions(
  record: Record<string, unknown>,
): Pick<TweetMedia, "width" | "height"> {
  const sizes = asRecord(record.sizes);
  const candidates = [
    sizes?.large,
    sizes?.medium,
    sizes?.small,
    sizes?.thumb,
    asRecord(record.original_info),
  ];
  for (const candidate of candidates) {
    const box = asRecord(candidate);
    const width = optionalPositiveInt(box?.w ?? box?.width);
    const height = optionalPositiveInt(box?.h ?? box?.height);
    if (width !== undefined && height !== undefined) return { width, height };
  }
  return {};
}

/**
 * Parses `extended_entities.media ?? entities.media` into archive-local media
 * references. The file name mirrors `data/tweets_media/<tweetId>/<fileName>`;
 * remote URLs are read but never stored.
 */
function parseTweetMedia(tweetId: string, source: Record<string, unknown>): TweetMedia[] {
  const entities = asRecord(source.extended_entities) ?? asRecord(source.entities);
  const list = entities?.media;
  if (!Array.isArray(list)) return [];

  const media: TweetMedia[] = [];
  for (const item of list) {
    const record = asRecord(item);
    if (!record) continue;
    const type = mediaType(record);
    const variant = type === "photo" ? undefined : mp4Variant(asRecord(record.video_info));
    const reference =
      variant?.url ??
      optionalText(record.media_url_https) ??
      optionalText(record.media_url);
    const basename = reference ? mediaBasename(reference) : undefined;
    if (!basename) continue;

    const entry: TweetMedia = {
      filename: tweetId ? `${tweetId}-${basename}` : basename,
      type,
    };
    const mime = variant?.contentType ?? optionalText(record.mime_type);
    if (mime !== undefined) entry.mime = mime;
    const { width, height } = mediaDimensions(record);
    if (width !== undefined) entry.width = width;
    if (height !== undefined) entry.height = height;
    const durationMs = optionalPositiveInt(asRecord(record.video_info)?.duration_millis);
    if (durationMs !== undefined) entry.durationMs = durationMs;
    media.push(entry);
  }
  return media;
}

/** Parses the modern `window.YTD.tweets.partN = [...]` archive format. */
export function parseTweetsJs(text: string): Tweet[] {
  const raw = parseJsonArray(text);
  const tweets: Tweet[] = [];
  for (const item of raw) {
    const wrapper = asRecord(item);
    const tweet = asRecord(wrapper?.tweet) ?? wrapper;
    if (!tweet) continue;
    const parsed = toTweet(
      tweet.full_text ?? tweet.text,
      tweet.id_str ?? tweet.id,
      tweet.created_at,
    );
    if (parsed) tweets.push(addJsMetadata(parsed, tweet));
  }
  return tweets;
}

/** Minimal RFC 4180 CSV row parser (quoted fields, embedded commas/newlines). */
export function parseCsvRows(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (quoted) {
      if (char === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          quoted = false;
        }
      } else {
        field += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else if (char !== "\r") {
      field += char;
    }
  }
  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}

/** Parses the legacy `data/tweets.csv` archive format. */
export function parseTweetsCsv(text: string): Tweet[] {
  const rows = parseCsvRows(text);
  if (rows.length < 2) return [];
  const header = rows[0].map((column) => column.trim());
  const idIndex = header.indexOf("tweet_id");
  const textIndex = header.indexOf("text");
  const timeIndex = header.indexOf("timestamp");
  if (textIndex < 0) return [];

  const tweets: Tweet[] = [];
  for (let i = 1; i < rows.length; i++) {
    const parsed = toTweet(
      rows[i][textIndex],
      idIndex >= 0 ? rows[i][idIndex] : String(i),
      rows[i][timeIndex],
    );
    if (parsed) tweets.push(parsed);
  }
  return tweets;
}

export function parseAccountJs(text: string): AccountInfo {
  const start = text.indexOf("[");
  if (start < 0) return {};
  try {
    const raw = JSON.parse(text.slice(start).trim().replace(/;$/, "")) as Array<{
      account?: {
        username?: string;
        name?: string;
        accountDisplayName?: string;
        accountId?: string;
        avatarImageUrl?: string;
      };
    }>;
    const account = raw[0]?.account;
    const result: AccountInfo = {};
    const username = optionalText(account?.username);
    const displayName = optionalText(account?.accountDisplayName ?? account?.name);
    const avatarUrl = optionalText(account?.avatarImageUrl);
    const accountId = optionalText(account?.accountId);
    if (username !== undefined) result.username = username;
    if (displayName !== undefined) result.displayName = displayName;
    if (avatarUrl !== undefined) result.avatarUrl = avatarUrl;
    if (accountId !== undefined) result.accountId = accountId;
    return result;
  } catch {
    return {};
  }
}

export function parseProfileJs(text: string): ProfileInfo {
  const start = text.indexOf("[");
  if (start < 0) return {};
  try {
    const raw = JSON.parse(text.slice(start).trim().replace(/;$/, "")) as Array<{
      profile?: {
        avatarMediaUrl?: string;
        headerMediaUrl?: string;
      };
    }>;
    const profile = raw[0]?.profile;
    const result: ProfileInfo = {};
    const avatarMediaUrl = optionalText(profile?.avatarMediaUrl);
    const headerMediaUrl = optionalText(profile?.headerMediaUrl);
    if (avatarMediaUrl !== undefined) result.avatarMediaUrl = avatarMediaUrl;
    if (headerMediaUrl !== undefined) result.headerMediaUrl = headerMediaUrl;
    return result;
  } catch {
    return {};
  }
}
