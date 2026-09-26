export type TweetMediaType = "photo" | "video" | "animated_gif";

/**
 * One media record attached to a tweet, resolved against the local archive.
 * Only file names are kept: no remote URL ever reaches UI-facing data.
 */
export interface TweetMedia {
  /** File name inside the archive's `data/tweets_media` folder. */
  filename: string;
  type: TweetMediaType;
  /** MIME type when the archive records one, e.g. `image/jpeg`, `video/mp4`. */
  mime?: string;
  width?: number;
  height?: number;
  /** Playback length in milliseconds for video and animated gif records. */
  durationMs?: number;
}

export interface Tweet {
  id: string;
  createdAt: number;
  text: string;
  username?: string;
  displayName?: string;
  avatarUrl?: string;
  isRetweet?: boolean;
  isReply?: boolean;
  favoriteCount?: number;
  retweetCount?: number;
  replyCount?: number;
  language?: string;
  /** Local media bundled with the archive; absent for text-only tweets. */
  media?: TweetMedia[];
}

export interface AccountInfo {
  username?: string;
  displayName?: string;
  avatarUrl?: string;
  avatarBlob?: Blob;
  accountId?: string;
}

export interface ArchiveSummary {
  tweets: Tweet[];
  account: AccountInfo;
  fileCount: number;
  skippedEntries: number;
}

export type AxisId = "EI" | "SN" | "TF" | "JP";

/** Minimal shape the aggregator needs from a Laya choice answer. */
export interface AxisAnswer {
  type: string;
  probabilities: Record<string, number>;
  confidence: number;
}

export interface Pole {
  code: string;
  label: string;
  description: string;
}

export interface AxisMeta {
  id: AxisId;
  name: string;
  positive: Pole;
  negative: Pole;
  neutral: string;
}

export interface AxisResult {
  id: AxisId;
  name: string;
  positive: Pole;
  negative: Pole;
  chosen: Pole;
  probability: number;
  positiveProbability: number;
  votes: number;
  neutralVotes: number;
  avgConfidence: number;
  marginBuckets?: number[];
  /**
   * Five buckets over the signed margin, from the negative pole (index 0) to
   * the positive pole (index 4). Unlike `marginBuckets`, which is unsigned and
   * only says how decisive a post was, this keeps the side each post leaned to,
   * so the card can draw a distribution along the axis itself.
   */
  signedBuckets?: number[];
  /**
   * Tweet answers on this axis that the model got wrong: a missing answer, an
   * answer that is not a two-way choice, or a probability that is missing or not
   * a finite number. These are excluded from the evidence, so a model failure
   * never counts as a vote and never inflates confidence.
   */
  invalidAnswers: number;
}

/** Per-axis fit of one candidate type against the accumulated evidence. */
export interface RankedAxisFit {
  id: AxisId;
  name: string;
  /** The pole this candidate type uses on the axis. */
  pole: Pole;
  /** Probability that this candidate's letter holds on the axis, 0..1. */
  matchProbability: number;
}

/** One of the sixteen MBTI types scored against the accumulated margins. */
export interface RankedType {
  type: string;
  /** 1-based rank, matching the position in `MbtiResult.ranking`. */
  rank: number;
  /** Normalized share of the sixteen-type distribution; the list sums to 1. */
  probability: number;
  /** Raw per-axis agreement product before normalization. */
  score: number;
  /** Axes whose letter differs from the top-ranked type. */
  differsFromTop: AxisId[];
  /** Count of flipped axes, i.e. Hamming distance from the top-ranked type. */
  flips: number;
  axes: RankedAxisFit[];
}

export interface MbtiResult {
  type: string;
  axes: AxisResult[];
  analyzed: number;
  usedTweets: number;
  /**
   * All sixteen types, best first. `ranking[0].type` always equals `type`, so
   * the runner-up is `ranking[1]` and third place is `ranking[2]`.
   */
  ranking: RankedType[];
  /**
   * Total tweet answers the model failed to answer with usable probabilities,
   * across all axes. Non-zero means part of the evidence is missing, not that the
   * remaining evidence contradicts `type`.
   */
  invalidAnswers: number;
  /**
   * True when no axis accumulated an effective vote, so `type` is the positive-pole
   * tie-break default and the sixteen-way ranking is uniform rather than measured.
   * Distinct from a model failure: check `invalidAnswers` to tell the two apart.
   */
  insufficientEvidence: boolean;
}
