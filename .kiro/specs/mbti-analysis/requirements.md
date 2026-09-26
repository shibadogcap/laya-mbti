# Feature Spec — MBTI estimation from an X archive

## Summary

Given one or more X archive ZIPs, estimate a four-letter MBTI type by running
each selected tweet through Laya typed decisions and aggregating the answers.

## Requirements (EARS)

### Archive ingestion

- WHEN a user selects one or more ZIP files, THE SYSTEM SHALL parse each archive
  inside a Web Worker without blocking the main thread.
- WHEN an archive contains media or unrelated entries, THE SYSTEM SHALL skip
  them and read only `data/tweets*.js`, `data/tweets*.csv`, and `data/account*.js`.
- WHEN the same tweet appears in more than one archive, THE SYSTEM SHALL keep a
  single copy keyed by tweet id.
- WHEN an archive is larger than available memory for a full extraction, THE
  SYSTEM SHALL decompress one entry at a time and retain only lightweight tweet
  records.
- IF a selected file is not a ZIP, THEN THE SYSTEM SHALL ignore it.

### Period selection

- WHEN an archive is parsed, THE SYSTEM SHALL show the tweet date range and allow
  the user to restrict analysis to a start/end period.
- WHEN the user picks a preset (all, last year, last 3 months, last month), THE
  SYSTEM SHALL set the period relative to the latest tweet in the archive.

### Pre-filtering

- WHEN preparing tweets for inference, THE SYSTEM SHALL drop retweets, replies
  (unless included), link-only tweets, and tweets shorter than the configured
  minimum content length.
- WHERE content length is measured, THE SYSTEM SHALL exclude URLs, mentions, and
  hashtags from the count.

### Judgment

- WHEN a tweet is analyzed, THE SYSTEM SHALL ask one two-way `choice` question
  per MBTI axis (positive pole vs negative pole).
- WHEN the model returns answers, THE SYSTEM SHALL record the chosen label, the
  full probability distribution, and the confidence for each question.
- WHEN each tweet finishes, THE SYSTEM SHALL emit it immediately so the UI can
  show a provisional estimate during the run.

### Execution backend

- WHEN inference starts, THE SYSTEM SHALL prefer the WebGPU execution provider
  for the encoder and only use the CPU (WASM) provider when WebGPU cannot run it.
- IF WebGPU is unavailable, cannot create a session, or fails on a batch that is
  already as small as a single tweet, THEN THE SYSTEM SHALL rebuild the model on
  the CPU (WASM) provider and analyze the selection again from the first tweet.
- WHEN a run restarts on another backend, THE SYSTEM SHALL discard every answer
  collected so far, so a partial result is never presented as a complete one.
- WHILE inference runs, THE SYSTEM SHALL report which backend is active, and on
  the CPU path say that it is considerably slower than WebGPU.
- IF the model files are already in Cache Storage, THEN THE SYSTEM SHALL rebuild
  without downloading them again.
- IF neither backend can analyze a single tweet, THEN THE SYSTEM SHALL report the
  failure instead of returning a partial aggregate.

### Aggregation

- WHEN aggregating an axis, THE SYSTEM SHALL compute the signed margin
  `P(positive) - P(negative)` for each tweet.
- WHEN a tweet's margin magnitude is below the configured threshold, THE SYSTEM
  SHALL count it as a retained (ambiguous) vote that does not move the axis.
- WHEN summing votes, THE SYSTEM SHALL accumulate positive and negative margins
  separately, so decisive tweets dominate and ambiguous tweets contribute little.
- WHEN an axis has no effective votes, THE SYSTEM SHALL fall back to an even
  split and report zero effective votes rather than guessing.
- WHEN aggregation completes, THE SYSTEM SHALL report the four-letter type plus,
  per axis, the winning probability, effective vote count, retained count, and
  average confidence.
- WHILE inference is running, THE SYSTEM SHALL display a throttled provisional
  aggregate.

### Sampling

- WHEN the filtered tweet count exceeds the configured maximum, THE SYSTEM SHALL
  sample evenly across the selected period instead of taking only the newest
  tweets.

### Privacy and honesty

- THE SYSTEM SHALL NOT transmit tweet text, account identifiers, or results to
  any server.
- THE SYSTEM SHALL display that the result is an unofficial estimate and not a
  diagnosis.

### Deployment

- WHEN the project builds for GitHub Pages, THE SYSTEM SHALL include the exported
  Laya model and ONNX Runtime Wasm assets and stay under 1 GB.
