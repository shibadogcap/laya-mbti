import { describe, expect, it } from "vitest";
import {
  TWEETS_MEDIA_RE,
  decodeEntities,
  mediaBasename,
  parseAccountJs,
  parseProfileJs,
  parseTweetsCsv,
  parseTweetsJs,
} from "../src/lib/parse.js";

describe("parseTweetsJs", () => {
  it("parses the modern window.YTD wrapper", () => {
    const text = `window.YTD.tweets.part0 = [
      { "tweet": { "id_str": "111", "created_at": "Mon Apr 22 10:00:00 +0000 2024", "full_text": "今日はカフェに行った。" } },
      { "tweet": { "id_str": "222", "created_at": "Tue Apr 23 10:00:00 +0000 2024", "full_text": "https://t.co/abc" } }
    ]`;
    const tweets = parseTweetsJs(text);
    expect(tweets).toHaveLength(2);
    expect(tweets[0]).toMatchObject({ id: "111", text: "今日はカフェに行った。" });
    expect(tweets[0].createdAt).toBe(
      Date.parse("Mon Apr 22 10:00:00 +0000 2024"),
    );
  });

  it("keeps optional modern metadata separate from older records", () => {
    const text = `window.YTD.tweets.part0 = [
      { "tweet": {
        "id_str": "111",
        "created_at": "Mon Apr 22 10:00:00 +0000 2024",
        "full_text": "今日はカフェに行った。",
        "favorite_count": 0,
        "retweet_count": 12,
        "reply_count": 3,
        "lang": "ja",
        "retweeted_status_id": "99",
        "in_reply_to_status_id": "100",
        "user": {
          "screen_name": "alice",
          "name": "Alice Example",
          "profile_image_url_https": "https://example.com/alice.jpg"
        }
      } },
      null,
      { "tweet": { "id_str": "222", "created_at": "Tue Apr 23 10:00:00 +0000 2024", "full_text": "古い形式" } }
    ]`;
    const tweets = parseTweetsJs(text);
    expect(tweets).toHaveLength(2);
    expect(tweets[0]).toStrictEqual({
      id: "111",
      createdAt: Date.parse("Mon Apr 22 10:00:00 +0000 2024"),
      text: "今日はカフェに行った。",
      username: "alice",
      displayName: "Alice Example",
      avatarUrl: "https://example.com/alice.jpg",
      favoriteCount: 0,
      retweetCount: 12,
      replyCount: 3,
      language: "ja",
      isRetweet: true,
      isReply: true,
    });
    expect(tweets[1]).toStrictEqual({
      id: "222",
      createdAt: Date.parse("Tue Apr 23 10:00:00 +0000 2024"),
      text: "古い形式",
    });
  });

  it("keeps counts that current archives serialize as strings", () => {
    const text = `window.YTD.tweets.part0 = [
      { "tweet": {
        "id_str": "111",
        "created_at": "Mon Apr 22 10:00:00 +0000 2024",
        "full_text": "いいね！",
        "favorite_count": "1234",
        "retweet_count": "0",
        "reply_count": "  7  "
      } },
      { "tweet": {
        "id_str": "222",
        "created_at": "Tue Apr 23 10:00:00 +0000 2024",
        "full_text": "壊れた値",
        "favorite_count": "many",
        "retweet_count": "-3",
        "reply_count": "1.5"
      } }
    ]`;
    const tweets = parseTweetsJs(text);
    expect(tweets[0]).toMatchObject({
      favoriteCount: 1234,
      retweetCount: 0,
      replyCount: 7,
    });
    expect(tweets[1]).not.toHaveProperty("favoriteCount");
    expect(tweets[1]).not.toHaveProperty("retweetCount");
    expect(tweets[1]).not.toHaveProperty("replyCount");
  });

  it("parses photo media from extended_entities", () => {
    const text = `window.YTD.tweets.part0 = [
      { "tweet": {
        "id_str": "1612000000000000000",
        "created_at": "Mon Apr 22 10:00:00 +0000 2024",
        "full_text": "写真です",
        "extended_entities": { "media": [
          {
            "id_str": "9001",
            "type": "photo",
            "media_url_https": "https://pbs.twimg.com/media/Ab-cdEfGh.jpg",
            "mime_type": "image/jpeg",
            "sizes": {
              "small": { "w": 340, "h": 226, "resize": "fit" },
              "large": { "w": 1200, "h": 797, "resize": "fit" }
            }
          },
          {
            "id_str": "9002",
            "type": "photo",
            "media_url_https": "https://pbs.twimg.com/media/second.png"
          }
        ] }
      } }
    ]`;
    const [tweet] = parseTweetsJs(text);
    expect(tweet.media).toEqual([
      {
        filename: "1612000000000000000-Ab-cdEfGh.jpg",
        type: "photo",
        mime: "image/jpeg",
        width: 1200,
        height: 797,
      },
      {
        filename: "1612000000000000000-second.png",
        type: "photo",
      },
    ]);
  });

  it("picks the highest-bitrate mp4 variant for video and animated gif", () => {
    const text = `window.YTD.tweets.part0 = [
      { "tweet": {
        "id_str": "333",
        "created_at": "Mon Apr 22 10:00:00 +0000 2024",
        "full_text": "動画",
        "extended_entities": { "media": [
          {
            "id_str": "9003",
            "type": "video",
            "media_url_https": "https://pbs.twimg.com/amplify_video/Poster.jpg",
            "video_info": {
              "aspect_ratio": [16, 9],
              "duration_millis": 30733,
              "variants": [
                { "bitrate": 832000, "content_type": "application/x-mpegURL", "url": "https://video.twimg.com/playlist.m3u8" },
                { "bitrate": 632000, "content_type": "video/mp4", "url": "https://video.twimg.com/low.mp4" },
                { "bitrate": 2176000, "content_type": "video/mp4", "url": "https://video.twimg.com/high-name.mp4" }
              ]
            }
          },
          {
            "id_str": "9004",
            "type": "animated_gif",
            "media_url_https": "https://pbs.twimg.com/tweet_video_thumb/GifName.gif",
            "video_info": {
              "duration_millis": 1500,
              "variants": [
                { "bitrate": 0, "content_type": "video/mp4", "url": "https://video.twimg.com/gif.mp4" }
              ]
            }
          },
          {
            "id_str": "9005",
            "type": "video",
            "media_url_https": "https://pbs.twimg.com/amplify_video/NoVariants.jpg",
            "video_info": { "variants": [] }
          }
        ] }
      } }
    ]`;
    const [tweet] = parseTweetsJs(text);
    expect(tweet.media).toEqual([
      {
        filename: "333-high-name.mp4",
        type: "video",
        mime: "video/mp4",
        durationMs: 30733,
      },
      {
        filename: "333-gif.mp4",
        type: "animated_gif",
        mime: "video/mp4",
        durationMs: 1500,
      },
      { filename: "333-NoVariants.jpg", type: "video" },
    ]);
  });

  it("falls back to entities.media and omits unknown records", () => {
    const text = `window.YTD.tweets.part0 = [
      { "tweet": {
        "id_str": "444",
        "created_at": "Mon Apr 22 10:00:00 +0000 2024",
        "full_text": "旧形式",
        "entities": { "media": [
          null,
          { "media_url_https": "https://pbs.twimg.com/media/legacy.jpg" }
        ] }
      } },
      { "tweet": {
        "id_str": "555",
        "created_at": "Mon Apr 22 10:00:00 +0000 2024",
        "full_text": "文字だけ"
      } }
    ]`;
    const tweets = parseTweetsJs(text);
    expect(tweets[0].media).toEqual([
      { filename: "444-legacy.jpg", type: "photo" },
    ]);
    expect(tweets[1]).not.toHaveProperty("media");
  });

  it("returns nothing when no array is present", () => {
    expect(parseTweetsJs("not json")).toEqual([]);
  });
});

describe("decodeEntities", () => {
  it("decodes the entities X escapes in full_text", () => {
    expect(decodeEntities("A&amp;B")).toBe("A&B");
    expect(decodeEntities("&lt;tag&gt;")).toBe("<tag>");
    expect(decodeEntities("&quot;quoted&quot;")).toBe('"quoted"');
    expect(decodeEntities("it&#39;s")).toBe("it's");
    expect(decodeEntities("&#x3042;")).toBe("あ");
  });

  it("decodes one layer only and leaves unknown entities alone", () => {
    expect(decodeEntities("&amp;lt;")).toBe("&lt;");
    expect(decodeEntities("&unknown;")).toBe("&unknown;");
    expect(decodeEntities("100% &amp; more")).toBe("100% & more");
  });

  it("keeps a full-width ampersand as written", () => {
    expect(decodeEntities("朝遅いとき＆今日がそれ")).toBe("朝遅いとき＆今日がそれ");
  });

  it("survives the whole parse path for an archived post", () => {
    const js = `window.YTD.tweets.part0 = [ ${JSON.stringify({
      tweet: {
        id_str: "1",
        created_at: "Wed Oct 10 20:19:24 +0000 2018",
        full_text: "R&amp;D と &lt;設計&gt;、Rs 5000 &amp;  #(escape)&#39;",
      },
    })} ];`;
    expect(parseTweetsJs(js)[0].text).toBe("R&D と <設計>、Rs 5000 &  #(escape)'");
  });
});

describe("parseTweetsCsv", () => {
  it("handles quoted commas and embedded newlines", () => {
    const csv = `tweet_id,in_reply_to_status_id,in_reply_to_user_id,timestamp,source,text,retweeted_status_id
9,,,2023-01-02 03:04:05 +0000,web,"Hello, ""world""!",,
10,,,2023-01-03 03:04:05 +0000,web,"line1
line2",,`;
    const tweets = parseTweetsCsv(csv);
    expect(tweets).toHaveLength(2);
    expect(tweets[0]).toStrictEqual({
      id: "9",
      createdAt: Date.parse("2023-01-02 03:04:05 +0000"),
      text: 'Hello, "world"!',
    });
    expect(tweets[1].text).toBe("line1\nline2");
    expect(tweets[1].id).toBe("10");
  });

  it("decodes escaped entities coming out of the archive", () => {
    const csv = `tweet_id,in_reply_to_status_id,in_reply_to_user_id,timestamp,source,text,retweeted_status_id
11,,,2023-01-04 03:04:05 +0000,web,"R&amp;D &lt;note&gt;",,`;
    expect(parseTweetsCsv(csv)[0].text).toBe("R&D <note>");
  });
});

describe("parseProfileJs", () => {
  it("extracts avatar and header media identifiers", () => {
    const text = `window.YTD.profile.part0 = [ { "profile": { "avatarMediaUrl": "https://pbs.twimg.com/profile_images/1/abc.jpg", "headerMediaUrl": "https://pbs.twimg.com/profile_banners/2/def.jpg" } } ];`;
    expect(parseProfileJs(text)).toEqual({
      avatarMediaUrl: "https://pbs.twimg.com/profile_images/1/abc.jpg",
      headerMediaUrl: "https://pbs.twimg.com/profile_banners/2/def.jpg",
    });
  });
});

describe("parseAccountJs", () => {
  it("extracts username, avatar, and account id", () => {
    const text = `window.YTD.account.part0 = [ { "account": { "username": "alice", "accountDisplayName": "Alice Example", "avatarImageUrl": "https://example.com/alice.jpg", "accountId": "42" } } ];`;
    expect(parseAccountJs(text)).toEqual({
      username: "alice",
      displayName: "Alice Example",
      avatarUrl: "https://example.com/alice.jpg",
      accountId: "42",
    });
  });

  it("is tolerant of malformed input", () => {
    expect(parseAccountJs("garbage")).toEqual({});
  });
});

describe("mediaBasename", () => {
  it("keeps hyphens in the file name and drops query strings", () => {
    expect(mediaBasename("https://pbs.twimg.com/media/Ab-cdEf.jpg?format=jpg&name=orig")).toBe(
      "Ab-cdEf.jpg",
    );
    expect(mediaBasename("data/tweets_media/1234/1234_Ab-cdEf.jpg")).toBe("1234_Ab-cdEf.jpg");
    expect(mediaBasename("https://pbs.twimg.com/media/")).toBeUndefined();
  });
});

describe("TWEETS_MEDIA_RE", () => {
  it("matches tweet media folders without swallowing tweets.js", () => {
    expect(TWEETS_MEDIA_RE.test("data/tweets_media/1234/1234_Ab.jpg")).toBe(true);
    expect(TWEETS_MEDIA_RE.test("archive/data/tweets_media/1234/1234_Ab.jpg")).toBe(true);
    expect(TWEETS_MEDIA_RE.test("data/tweets.js")).toBe(false);
    expect(TWEETS_MEDIA_RE.test("data/tweets-media/1234.jpg")).toBe(false);
  });
});
