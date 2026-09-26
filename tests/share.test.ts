import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { SHARE_HASHTAG, SITE_URL } from "../src/App.js";

const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");

describe("share metadata", () => {
  it("points canonical and og:url at the pinned site url", () => {
    expect(html).toContain(`<link rel="canonical" href="${SITE_URL}/" />`);
    expect(html).toContain(`<meta property="og:url" content="${SITE_URL}/" />`);
  });

  it("advertises an og image that exists", () => {
    const tag = html.match(/<meta property="og:image" content="([^"]+)" \/>/);
    expect(tag).not.toBeNull();
    const url = new URL(tag![1]);
    expect(url.pathname).toBe("/ogp.png");
    expect(existsSync(new URL("../public/ogp.png", import.meta.url))).toBe(true);
    expect(html).toContain(`<meta name="twitter:image" content="${url.href}" />`);
  });

  it("uses an underscored hashtag in the share text", () => {
    expect(SHARE_HASHTAG).toBe("#laya_mbti");
    expect(SHARE_HASHTAG).not.toContain("-");
  });

  it("has a large-image card for the share preview", () => {
    expect(html).toContain('<meta name="twitter:card" content="summary_large_image" />');
  });
});
