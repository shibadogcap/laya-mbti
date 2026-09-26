/// <reference lib="webworker" />
import { BlobReader, BlobWriter, TextWriter, ZipReader } from "@zip.js/zip.js";
import {
  ACCOUNT_RE,
  PROFILE_MEDIA_RE,
  PROFILE_RE,
  TWEETS_MEDIA_RE,
  TWEETS_RE,
  mediaBasename,
  parseAccountJs,
  parseProfileJs,
  parseTweetsCsv,
  parseTweetsJs,
} from "../lib/parse.js";
import type { AccountInfo, Tweet } from "../lib/types.js";

type Request =
  | { type: "start"; files: File[] }
  | { type: "cancel" }
  | { type: "readMedia"; id: number; filename: string };

type Response =
  | { type: "status"; message: string }
  | { type: "fileStart"; index: number; total: number; name: string }
  | {
      type: "progress";
      index: number;
      total: number;
      name: string;
      loaded: number;
      totalBytes: number;
    }
  | {
      type: "done";
      tweets: Tweet[];
      account: AccountInfo;
      fileCount: number;
      skippedEntries: number;
    }
  | { type: "media"; id: number; blob: Blob | null }
  | { type: "error"; message: string };

interface ZipEntry {
  filename: string;
  directory?: boolean;
  uncompressedSize?: number;
  getData?: unknown;
}

/** Tweet media above this size is never materialized, keeping memory flat. */
const MEDIA_BLOB_LIMIT = 1_500_000;
const IMAGE_FILE_RE = /\.(?:png|jpe?g|gif|webp|avif|bmp)$/i;
const VIDEO_FILE_RE = /\.(?:mp4|m4v|mov|webm)$/i;

const ctx = self as unknown as DedicatedWorkerGlobalScope;
let controller: AbortController | null = null;
/** Tweet media entries keyed by exact archive file name, kept after `done`. */
const mediaEntries = new Map<string, ZipEntry>();

function post(message: Response): void {
  ctx.postMessage(message);
}

async function readEntry(
  entry: ZipEntry,
  index: number,
  total: number,
  signal: AbortSignal,
): Promise<string> {
  const getData = entry.getData as (
    writer: TextWriter,
    options: {
      signal: AbortSignal;
      onprogress?: (progress: { loaded: number; total?: number }) => void;
    },
  ) => Promise<string>;
  return getData(new TextWriter(), {
    signal,
    onprogress: (progress) =>
      post({
        type: "progress",
        index,
        total,
        name: entry.filename,
        loaded: progress.loaded,
        totalBytes: progress.total ?? entry.uncompressedSize ?? 0,
      }),
  });
}

function mimeForFilename(filename: string): string {
  if (/\.png$/i.test(filename)) return "image/png";
  if (/\.gif$/i.test(filename)) return "image/gif";
  if (/\.webp$/i.test(filename)) return "image/webp";
  if (/\.avif$/i.test(filename)) return "image/avif";
  return "image/jpeg";
}

type BlobEntryReader = (
  writer: BlobWriter,
  options: {
    signal: AbortSignal;
    onprogress?: (progress: { loaded: number; total?: number }) => void;
  },
) => Promise<Blob>;

async function extractBlob(
  entry: ZipEntry,
  signal: AbortSignal,
  onprogress?: (progress: { loaded: number; total?: number }) => void,
): Promise<Blob> {
  const getData = entry.getData as BlobEntryReader;
  return getData(new BlobWriter(mimeForFilename(entry.filename)), { signal, onprogress });
}

function readBinaryEntry(
  entry: ZipEntry,
  index: number,
  total: number,
  signal: AbortSignal,
): Promise<Blob> {
  return extractBlob(entry, signal, (progress) =>
    post({
      type: "progress",
      index,
      total,
      name: entry.filename,
      loaded: progress.loaded,
      totalBytes: progress.total ?? entry.uncompressedSize ?? 0,
    }),
  );
}

function mediaStem(value: string | undefined): string | undefined {
  if (!value) return undefined;
  const filename = mediaBasename(value);
  if (!filename) return undefined;
  const stem = filename.split(".")[0];
  return stem ? stem.toLowerCase() : undefined;
}

function mediaMatches(filename: string, url: string | undefined): boolean {
  const expected = mediaStem(url);
  if (!expected) return false;
  const candidate = mediaStem(filename);
  return (
    candidate === expected ||
    (candidate !== undefined &&
      (candidate.endsWith(`-${expected}`) || candidate.endsWith(`_${expected}`)))
  );
}

/**
 * Indexes a tweet media entry under its exact archive path and its bare file
 * name, so lookups never depend on guessing an id out of a hyphenated name.
 */
function indexMediaEntry(entry: ZipEntry): void {
  if (!mediaEntries.has(entry.filename)) mediaEntries.set(entry.filename, entry);
  const name = mediaBasename(entry.filename);
  if (name && !mediaEntries.has(name)) mediaEntries.set(name, entry);
}

/** Serves one bundled image; returns null for anything else, mp4 included. */
async function readMediaBlob(filename: string, signal: AbortSignal): Promise<Blob | null> {
  if (filename.length === 0 || filename.includes("..")) return null;
  if (VIDEO_FILE_RE.test(filename) || !IMAGE_FILE_RE.test(filename)) return null;
  const entry = mediaEntries.get(filename);
  if (!entry) return null;
  if ((entry.uncompressedSize ?? 0) > MEDIA_BLOB_LIMIT) return null;
  const blob = await extractBlob(entry, signal);
  return blob.size <= MEDIA_BLOB_LIMIT ? blob : null;
}

async function process(files: File[]): Promise<void> {
  controller = new AbortController();
  const { signal } = controller;

  const tweetsById = new Map<string, Tweet>();
  let account: AccountInfo = {};
  let skippedEntries = 0;
  let fileCount = 0;

  for (let index = 0; index < files.length; index++) {
    const file = files[index];
    fileCount++;
    post({ type: "fileStart", index, total: files.length, name: file.name });

    const reader = new ZipReader(new BlobReader(file), {
      useWebWorkers: false,
    });
    try {
      const entries = (await reader.getEntries()) as unknown as ZipEntry[];
      const profileMediaEntries: ZipEntry[] = [];
      let profileAvatarUrl: string | undefined;
      for (const entry of entries) {
        if (signal.aborted) return;
        if (entry.directory) continue;

        if (TWEETS_RE.test(entry.filename)) {
          post({
            type: "status",
            message: `${file.name}: ${entry.filename} を読み込み中`,
          });
          const text = await readEntry(entry, index, files.length, signal);
          const parsed = entry.filename.toLowerCase().endsWith(".csv")
            ? parseTweetsCsv(text)
            : parseTweetsJs(text);
          for (const tweet of parsed) {
            if (tweet.id) tweetsById.set(tweet.id, tweet);
          }
        } else if (ACCOUNT_RE.test(entry.filename)) {
          const text = await readEntry(entry, index, files.length, signal);
          account = { ...parseAccountJs(text), ...account };
          profileAvatarUrl ??= account.avatarUrl;
        } else if (PROFILE_RE.test(entry.filename)) {
          const text = await readEntry(entry, index, files.length, signal);
          profileAvatarUrl = parseProfileJs(text).avatarMediaUrl ?? profileAvatarUrl;
        } else if (PROFILE_MEDIA_RE.test(entry.filename)) {
          profileMediaEntries.push(entry);
        } else if (TWEETS_MEDIA_RE.test(entry.filename)) {
          indexMediaEntry(entry);
        } else {
          skippedEntries++;
        }
      }

      const avatarEntry =
        profileMediaEntries.find((entry) => mediaMatches(entry.filename, profileAvatarUrl)) ??
        profileMediaEntries.find((entry) => /avatar/i.test(entry.filename));
      if (avatarEntry) {
        try {
          const avatarBlob = await readBinaryEntry(avatarEntry, index, files.length, signal);
          if (avatarBlob.size <= 5_000_000) account.avatarBlob = avatarBlob;
        } catch {
        }
      }
    } finally {
      await reader.close();
    }
  }

  const tweets = Array.from(tweetsById.values()).sort(
    (a, b) => a.createdAt - b.createdAt,
  );
  post({ type: "done", tweets, account, fileCount, skippedEntries });
}

ctx.onmessage = (event: MessageEvent<Request>) => {
  const data = event.data;
  if (data.type === "cancel") {
    controller?.abort();
    mediaEntries.clear();
    return;
  }
  if (data.type === "readMedia") {
    const signal = controller?.signal ?? new AbortController().signal;
    readMediaBlob(data.filename, signal)
      .then((blob) => post({ type: "media", id: data.id, blob }))
      .catch(() => post({ type: "media", id: data.id, blob: null }));
    return;
  }
  if (data.type === "start") {
    process(data.files).catch((error: unknown) => {
      post({
        type: "error",
        message: error instanceof Error ? error.message : String(error),
      });
    });
  }
};
