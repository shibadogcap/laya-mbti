# Product Steering — laya-mbti

## What this is

`laya-mbti` is a static, client-only web app that estimates MBTI tendencies from
an X (Twitter) archive. The user drops one or more archive ZIPs, picks a period,
and the app runs every selected tweet through the Laya typed-decision model
(official `laya-ts`) to obtain per-tweet judgments on the four MBTI axes, then
aggregates them into a four-letter type.

## Who it is for

- People curious about how their own writing style maps onto MBTI language.
- Users who care about privacy and do not want to upload their archive anywhere.
- Developers looking for a worked example of on-device LLM-style inference.

## Principles

1. **Client-only.** No tweet text, account name, or result ever leaves the
   browser. There is no backend and no analytics.
2. **Honest uncertainty.** Every axis shows confidence, effective vote counts,
   and how many tweets were rejected. Never present a guess as a diagnosis.
3. **Graceful with huge inputs.** Archives of several GB, split across multiple
   ZIPs, must work without freezing the UI or exhausting memory.
4. **Reproducible builds.** The model is exported from a pinned checkpoint in
   CI; nothing large is committed to git.

## Non-goals

- Clinical or psychological assessment. The result is a stylistic estimate.
- Server-side inference, accounts, sharing backends, or storage of user data.
- Supporting non-archive inputs (live API scraping is out of scope).

## Disclaimer

The UI must always state that the result is an unofficial, statistical estimate
and is not a diagnosis, and that the app is not affiliated with X Corp.
