#!/usr/bin/env node
// Pre-deployment integrity and size check for the GitHub Pages artifact.
import { readFile, readdir, stat } from "node:fs/promises";
import { join } from "node:path";

const DIST = process.argv[2] ?? "dist";
const SIZE_BUDGET = 1_000_000_000;

const REQUIRED = [
  "index.html",
  "sw.js",
  "ort/ort-wasm-simd-threaded.jsep.mjs",
  "ort/ort-wasm-simd-threaded.jsep.wasm",
  "models/laya/encoder.onnx",
  "models/laya/head.onnx",
  "models/laya/tokenizer.json",
  "models/laya/rl_agent_config.json",
].map((file) => join(DIST, file));

async function walk(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) files.push(...(await walk(path)));
    else files.push(path);
  }
  return files;
}

async function main() {
  const missing = [];
  for (const file of REQUIRED) {
    try {
      const info = await stat(file);
      if (info.size === 0) missing.push(`${file} (empty)`);
    } catch {
      missing.push(file);
    }
  }
  if (missing.length > 0) {
    throw new Error(`Missing required assets:\n  ${missing.join("\n  ")}`);
  }

  // A broken or fp32 export would blow past these floors/sizes.
  const encoder = await stat(join(DIST, "models", "laya", "encoder.onnx"));
  if (encoder.size < 400_000_000) {
    throw new Error(
      `encoder.onnx is only ${encoder.size} bytes; expected a ~616 MB fp16 export`,
    );
  }
  const cfg = JSON.parse(
    await readFile(join(DIST, "models", "laya", "rl_agent_config.json"), "utf8"),
  );
  if (typeof cfg.max_len !== "number" || typeof cfg.head_max_len !== "number") {
    throw new Error("rl_agent_config.json is missing max_len/head_max_len");
  }

  const files = await walk(DIST);
  let total = 0;
  for (const file of files) total += (await stat(file)).size;

  console.log(`Files: ${files.length}`);
  console.log(`Total: ${total.toLocaleString()} bytes`);
  if (total > SIZE_BUDGET) {
    throw new Error(
      `Deployed site exceeds ${SIZE_BUDGET.toLocaleString()} byte budget ` +
        `(GitHub Pages limit is 1 GB)`,
    );
  }

  await assertNoAnnotationTool(DIST, files);
  console.log("validate-pages: OK");
}

/**
 * The review overlay is dev-only and must never ship: it injects a document
 * level listener that swallows clicks on the real UI, so a leak would break the
 * deployed page for every visitor, not just add a stray widget.
 */
const ANNOTATION_MARKERS = [
  "agent-ui-annotation",
  "agent-ui-annotation:",
  "AGENT-UI-ANNOTATION",
  "annotation:create",
  "mountAgentUiAnnotation",
];

async function assertNoAnnotationTool(dist, files) {
  const suspects = files.filter(
    (file) => /\.(js|mjs|css|html|json|map)$/i.test(file) && !file.includes(`${dist}/models/`),
  );
  const leaks = [];
  for (const file of suspects) {
    const body = await readFile(file, "utf8");
    for (const marker of ANNOTATION_MARKERS) {
      if (body.includes(marker)) leaks.push(`${file.replace(dist, "dist")} -> ${marker}`);
    }
  }
  if (leaks.length > 0) {
    throw new Error(
      `Dev-only annotation tool leaked into the build:\n  ${leaks.join("\n  ")}`,
    );
  }
  console.log(`annotation tool: absent from ${suspects.length} text assets`);
}

main().catch((error) => {
  console.error(`validate-pages: ${error.message}`);
  process.exit(1);
});
