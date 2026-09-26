// Persistent cache for the large, immutable model and ONNX Runtime assets.
//
// laya-web fetches these with a plain `fetch()` and no explicit cache, so the
// browser only keeps them for the HTTP cache lifetime. Intercepting here stores
// them in Cache Storage, which makes reloads and extra parallel workers cheap.
//
// Scope note: service workers require a secure context, so this works on
// https:// (GitHub Pages) and http://localhost, but not on a plain LAN IP.

const CACHE = "laya-assets-v1";
const REMOTE_MODEL = "https://r4ai.github.io/laya-web/models/laya/";

function isModelAsset(url, requestUrl) {
  if (requestUrl.startsWith(REMOTE_MODEL)) return true;
  if (url.origin !== self.location.origin) return false;
  return url.pathname.includes("/models/laya/") || url.pathname.includes("/ort/");
}

self.addEventListener("install", () => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (!isModelAsset(url, event.request.url)) return;
  event.respondWith(cacheFirst(event.request));
});

async function cacheFirst(request) {
  const cache = await caches.open(CACHE);
  const hit = await cache.match(request);
  if (hit) return hit;

  const response = await fetch(request);
  if (response.ok) {
    try {
      await cache.put(request, response.clone());
    } catch {
      // Quota or opaque-response failures must not break the download.
    }
  }
  return response;
}
