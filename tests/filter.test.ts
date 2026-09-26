import { describe, expect, it } from "vitest";
import { contentLength, preFilter, sampleTweets } from "../src/lib/filter.js";
import type { Tweet } from "../src/lib/types.js";

const tweet = (id: string, text: string): Tweet => ({
  id,
  createdAt: Number(id),
  text,
});

describe("contentLength", () => {
  it("ignores URLs, mentions and hashtags", () => {
    expect(contentLength("https://t.co/abc @bob #tag")).toBe(0);
    expect(contentLength("こんにちは https://t.co/abc")).toBe(5);
  });
});

describe("preFilter", () => {
  it("drops link-only, short, retweet and reply tweets", () => {
    const { kept, stats } = preFilter(
      [
        tweet("1", "https://t.co/abc"),
        tweet("2", "おはよう"),
        tweet("3", "RT @bob: これはすごい"),
        tweet("4", "@bob そうだね、本当にそう思うよ"),
        tweet("5", "今日は新しいカフェに行ってゆっくり本を読んだ"),
      ],
      { minLength: 10, includeReplies: false, excludeRetweets: true },
    );
    expect(kept.map((t) => t.id)).toEqual(["5"]);
    expect(stats.linkOnly).toBe(1);
    expect(stats.tooShort).toBe(1);
    expect(stats.retweets).toBe(1);
    expect(stats.replies).toBe(1);
    expect(stats.duplicateTweets).toBe(0);
  });

  it("deduplicates formatting variants and keeps the earliest archive tweet", () => {
    const { kept, stats } = preFilter(
      [
        tweet("10", "I really enjoyed this café, truly! https://t.co/abc @alice #food"),
        tweet("2", "i   REALLY enjoyed this cafe truly @bob https://t.co/def"),
        tweet("3", "I REALLY enjoyed this café, truly!!!"),
        tweet("4", "A different note about architecture and cities"),
      ],
      { minLength: 10, includeReplies: false, excludeRetweets: true },
    );

    expect(kept.map((t) => t.id)).toEqual(["10", "4"]);
    expect(stats.total).toBe(4);
    expect(stats.tooShort).toBe(0);
    expect(stats.linkOnly).toBe(0);
    expect(stats.retweets).toBe(0);
    expect(stats.replies).toBe(0);
    expect(stats.duplicateTweets).toBe(2);
    expect(stats.kept).toBe(2);
  });

  it("keeps metadata retweets separate from duplicate posts", () => {
    const original: Tweet = {
      ...tweet("1", "同じ投稿本文"),
      isRetweet: true,
    };
    const { kept, stats } = preFilter(
      [original, tweet("2", "同じ投稿本文"), tweet("3", "同じ投稿本文")],
      { minLength: 1, includeReplies: false, excludeRetweets: true },
    );
    expect(kept.map((item) => item.id)).toEqual(["2"]);
    expect(stats.retweets).toBe(1);
    expect(stats.duplicateTweets).toBe(1);
  });
});

describe("sampleTweets", () => {
  it("returns everything when under the limit", () => {
    const tweets = [tweet("1", "a"), tweet("2", "b")];
    expect(sampleTweets(tweets, 5)).toBe(tweets);
  });

  it("evenly samples across the list", () => {
    const tweets = Array.from({ length: 10 }, (_, i) => tweet(String(i), "x"));
    const sampled = sampleTweets(tweets, 4);
    expect(sampled).toHaveLength(4);
    expect(sampled[0].id).toBe("0");
    expect(sampled.at(-1)?.id).toBe("7");
  });

  it("returns the sample in chronological order even if the input is not sorted", () => {
    const tweets = [
      { ...tweet("3", "x"), createdAt: 30 },
      { ...tweet("1", "x"), createdAt: 10 },
      { ...tweet("5", "x"), createdAt: 50 },
      { ...tweet("2", "x"), createdAt: 20 },
      { ...tweet("4", "x"), createdAt: 40 },
    ];
    expect(sampleTweets(tweets, 3).map((item) => item.createdAt)).toEqual([10, 20, 40]);
  });
});
