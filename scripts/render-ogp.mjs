#!/usr/bin/env node
/**
 * Renders `ogp.html` to `public/ogp.png` at 1200x630 with headless Chrome.
 *
 * The dev server is used on purpose: the page pulls the webfont out of
 * node_modules, which a static server rooted at `dist` cannot resolve.
 */
import { spawn, spawnSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const PORT = 4188;
const CHROME = [
  "google-chrome",
  "chromium",
  "chromium-browser",
  "/usr/bin/google-chrome",
].find((bin) => {
  const probe = spawnSync("which", [bin], { encoding: "utf8" });
  return probe.status === 0 || existsSync(bin);
});
if (!CHROME) {
  console.error("Chrome/Chromium not found; cannot render the OG image.");
  process.exit(1);
}

const profile = join(tmpdir(), "laya-ogp-profile");
rmSync(profile, { recursive: true, force: true });
mkdirSync(profile, { recursive: true });
const out = join(tmpdir(), "ogp-shot.png");

const server = spawn(
  "pnpm",
  ["exec", "vite", "--host", "127.0.0.1", "--port", String(PORT), "--strictPort"],
  { stdio: "ignore" },
);

const shutdown = () => {
  server.kill("SIGTERM");
  rmSync(profile, { recursive: true, force: true });
};
process.on("exit", shutdown);

try {
  for (let attempt = 0; attempt < 30; attempt++) {
    const probe = spawnSync("curl", ["-sf", "-o", "/dev/null", `http://127.0.0.1:${PORT}/`]);
    if (probe.status === 0) break;
    await sleep(500);
  }

  const shot = spawnSync(
    CHROME,
    [
      "--headless=new",
      "--disable-gpu",
      "--hide-scrollbars",
      "--no-sandbox",
      "--force-device-scale-factor=1",
      `--user-data-dir=${profile}`,
      `--window-size=1200,630`,
      `--screenshot=${out}`,
      `http://127.0.0.1:${PORT}/ogp.html`,
    ],
    { stdio: "ignore" },
  );
  if (shot.status !== 0 || !existsSync(out)) {
    throw new Error("Chrome did not produce a screenshot");
  }
  // /tmp can be a different filesystem, so copy instead of rename.
  copyFileSync(out, "public/ogp.png");
  console.log("public/ogp.png written (1200x630)");
} finally {
  shutdown();
}
