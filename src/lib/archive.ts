import type { AccountInfo, Tweet } from "./types.js";

export interface ArchiveProgress {
  phase: "file" | "status" | "progress";
  fileIndex: number;
  fileTotal: number;
  fileName: string;
  loaded: number;
  totalBytes: number;
  message: string;
}

export interface ArchiveSummary {
  tweets: Tweet[];
  account: AccountInfo;
  fileCount: number;
  skippedEntries: number;
}

export interface ArchiveTask {
  promise: Promise<ArchiveSummary>;
  cancel: () => void;
  /**
   * Reads one bundled tweet image out of the parsed archive, by the exact
   * `TweetMedia.filename`. Resolves null when the entry is missing, larger than
   * 1.5 MB, not an image, or the task is no longer running. Never fetches
   * anything from the network, and never returns video.
   */
  readMedia: (filename: string) => Promise<Blob | null>;
}

type WorkerResponse =
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

/**
 * Streams one or more X archive ZIPs in a worker and returns the merged tweets.
 * The worker stays alive after `done` so bundled tweet images can still be read
 * on demand; unrelated entries are skipped, so archives over 1 GB are fine.
 */
export function readArchive(
  files: File[],
  onProgress: (progress: ArchiveProgress) => void,
): ArchiveTask {
  const worker = new Worker(
    new URL("../workers/archive.worker.ts", import.meta.url),
    { type: "module" },
  );

  const pendingMedia = new Map<number, (blob: Blob | null) => void>();
  let running = true;
  let settled = false;
  let nextMediaId = 1;
  let rejectPromise: ((error: Error) => void) | null = null;

  const stop = (): void => {
    running = false;
    worker.terminate();
    for (const settle of pendingMedia.values()) settle(null);
    pendingMedia.clear();
  };

  const promise = new Promise<ArchiveSummary>((resolve, reject) => {
    rejectPromise = reject;
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const data = event.data;
      switch (data.type) {
        case "status":
          onProgress({
            phase: "status",
            fileIndex: 0,
            fileTotal: files.length,
            fileName: "",
            loaded: 0,
            totalBytes: 0,
            message: data.message,
          });
          break;
        case "fileStart":
          onProgress({
            phase: "file",
            fileIndex: data.index,
            fileTotal: data.total,
            fileName: data.name,
            loaded: 0,
            totalBytes: 0,
            message: `${data.name} を解析中`,
          });
          break;
        case "progress":
          onProgress({
            phase: "progress",
            fileIndex: data.index,
            fileTotal: data.total,
            fileName: data.name,
            loaded: data.loaded,
            totalBytes: data.totalBytes,
            message: data.name,
          });
          break;
        case "done":
          settled = true;
          resolve({
            tweets: data.tweets,
            account: data.account,
            fileCount: data.fileCount,
            skippedEntries: data.skippedEntries,
          });
          break;
        case "media":
          pendingMedia.get(data.id)?.(data.blob);
          pendingMedia.delete(data.id);
          break;
        case "error":
          settled = true;
          stop();
          reject(new Error(data.message));
          break;
      }
    };
    worker.onerror = (event) => {
      settled = true;
      stop();
      reject(new Error(event.message || "アーカイブ解析ワーカーでエラー"));
    };
    worker.postMessage({ type: "start", files });
  });

  return {
    promise,
    cancel: () => {
      if (!running) return;
      worker.postMessage({ type: "cancel" });
      stop();
      if (!settled) {
        settled = true;
        rejectPromise?.(new Error("アーカイブ解析をキャンセルしました"));
      }
    },
    readMedia: (filename: string) =>
      new Promise<Blob | null>((resolve) => {
        if (!running) {
          resolve(null);
          return;
        }
        const id = nextMediaId++;
        pendingMedia.set(id, resolve);
        worker.postMessage({ type: "readMedia", id, filename });
      }),
  };
}
