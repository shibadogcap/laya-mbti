import type { Tweet } from "./types.js";

export interface PreFilterOptions {
  minLength: number;
  includeReplies: boolean;
  excludeRetweets: boolean;
}

export interface FilterStats {
  total: number;
  tooShort: number;
  linkOnly: number;
  retweets: number;
  replies: number;
  duplicateTweets: number;
  kept: number;
}

const URL_RE = /https?:\/\/\S+/gi;
const MENTION_RE = /@[A-Za-z0-9_]+/g;
const HASHTAG_RE = /#\S+/g;
const FINGERPRINT_URL_RE = /(?:https?:\/\/|www\.)\S+/gi;
const FINGERPRINT_PUNCTUATION_RE = /\p{P}/gu;
const FINGERPRINT_MARKS_RE = /\p{M}/gu;
const FINGERPRINT_FORMAT_RE = /\p{Cf}/gu;
const RT_RE = /^RT\s+@[A-Za-z0-9_]+:/;
const REPLY_RE = /^@[A-Za-z0-9_]+\s/;

function contentFingerprint(text: string): string {
  return text
    .normalize("NFKD")
    .toLowerCase()
    .replace(FINGERPRINT_FORMAT_RE, "")
    .replace(FINGERPRINT_URL_RE, " ")
    .replace(MENTION_RE, " ")
    .replace(HASHTAG_RE, " ")
    .replace(FINGERPRINT_PUNCTUATION_RE, " ")
    .replace(FINGERPRINT_MARKS_RE, "")
    .replace(/\s+/g, " ")
    .trim();
}

/** Removes URLs, mentions and hashtags to test whether any real prose remains. */
export function contentLength(text: string): number {
  return text
    .replace(URL_RE, "")
    .replace(MENTION_RE, "")
    .replace(HASHTAG_RE, "")
    .replace(/\s+/g, " ")
    .trim().length;
}

export function isRetweet(text: string, tweet?: Pick<Tweet, "isRetweet">): boolean {
  return tweet?.isRetweet === true || RT_RE.test(text.trim());
}

export function isReply(text: string, tweet?: Pick<Tweet, "isReply">): boolean {
  return tweet?.isReply === true || REPLY_RE.test(text.trim());
}

/**
 * Cheap, answer-independent filter applied before inference. Removes tweets that
 * can never carry personality signal so we do not spend model passes on them.
 */
export function preFilter(
  tweets: Tweet[],
  options: PreFilterOptions,
): { kept: Tweet[]; stats: FilterStats } {
  const stats: FilterStats = {
    total: tweets.length,
    tooShort: 0,
    linkOnly: 0,
    retweets: 0,
    replies: 0,
    duplicateTweets: 0,
    kept: 0,
  };
  const kept: Tweet[] = [];
  const fingerprints = new Set<string>();

  for (const tweet of tweets) {
    const text = tweet.text.trim();
    if (options.excludeRetweets && isRetweet(text, tweet)) {
      stats.retweets++;
      continue;
    }
    if (!options.includeReplies && isReply(text, tweet)) {
      stats.replies++;
      continue;
    }
    const content = contentLength(text);
    if (content === 0) {
      stats.linkOnly++;
      continue;
    }
    if (content < options.minLength) {
      stats.tooShort++;
      continue;
    }
    const fingerprint = contentFingerprint(text);
    if (fingerprints.has(fingerprint)) {
      stats.duplicateTweets++;
      continue;
    }
    fingerprints.add(fingerprint);
    kept.push(tweet);
  }

  stats.kept = kept.length;
  return { kept, stats };
}

export function filterByPeriod(
  tweets: Tweet[],
  from: number | null,
  to: number | null,
): Tweet[] {
  if (from === null && to === null) return tweets;
  return tweets.filter((tweet) => {
    if (tweet.createdAt === 0) return false;
    if (from !== null && tweet.createdAt < from) return false;
    if (to !== null && tweet.createdAt > to) return false;
    return true;
  });
}

/**
 * Evenly samples up to `limit` tweets across the whole list, so a capped run
 * covers the entire selected period instead of only the newest tweets. The
 * result is chronological (oldest first) whatever order the input arrived in.
 */
export function sampleTweets(tweets: Tweet[], limit: number): Tweet[] {
  if (limit <= 0 || tweets.length <= limit) return tweets;
  const chronological = [...tweets].sort((a, b) => a.createdAt - b.createdAt);
  const step = chronological.length / limit;
  const sampled: Tweet[] = [];
  for (let i = 0; i < limit; i++) {
    sampled.push(chronological[Math.floor(i * step)]);
  }
  return sampled;
}
