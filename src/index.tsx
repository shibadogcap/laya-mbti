import { render } from "solid-js/web";
import "@fontsource-variable/noto-sans-jp";
import App from "./App.js";
import "./styles.css";

const root = document.getElementById("root");
if (root) render(() => <App />, root);

if (import.meta.env.DEV) {
  void import("./annotation-dev.js")
    .then(({ mountAgentUiAnnotation }) => mountAgentUiAnnotation())
    .catch(() => undefined);
}

// Cache the ~934 MB model and ONNX Runtime assets in Cache Storage so reloads
// and additional inference workers do not re-download them. Requires a secure
// context (https or localhost).
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    void navigator.serviceWorker
      .register(new URL("sw.js", document.baseURI).href)
      .catch(() => {
        // Caching is an optimization; ignore failures (e.g. insecure origin).
      });
  });
}
