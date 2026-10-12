import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import { readdir, rm } from "node:fs/promises";
import { join } from "node:path";

// COOP/COEP are required for wasm threads, and must match app/public/_headers so dev
// and production behave alike. require-corp, not credentialless: Safari does not
// support credentialless and silently loses isolation there. The cross-origin weights
// send no CORP header and do not need to -- COEP runs the CORP check only on no-cors
// loads, and these arrive through fetch() in cors mode.
const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

/** Two kinds of dead weight in dist:
 *  - public/models is a ~900MB dev convenience once VITE_MODELS_BASE points at a CDN;
 *  - the bundler emits ORT .wasm into assets/ (including a 26.9MB asyncify variant we
 *    never use), while env.wasm.wasmPaths always resolves to /ort/. The asyncify one
 *    also exceeds Cloudflare Pages' 25MiB per-file cap, so leaving it fails the deploy.
 */
function trimDist(base: string | undefined) {
  return {
    name: "trim-dist",
    apply: "build" as const,
    async closeBundle() {
      if (base) {
        await rm("dist/models", { recursive: true, force: true });
        console.log(`weights served from ${base}; removed dist/models`);
      }
      const stray = (await readdir("dist/assets")).filter((f) => f.endsWith(".wasm"));
      for (const f of stray) await rm(join("dist/assets", f));
      if (stray.length) console.log(`removed ${stray.length} unused ORT wasm from dist/assets`);
    },
  };
}

export default defineConfig(({ mode }) => {
  const base = loadEnv(mode, process.cwd(), "VITE_").VITE_MODELS_BASE;
  return {
    plugins: [react(), trimDist(base)],
    server: { headers: isolation },
    preview: { headers: isolation },
    optimizeDeps: { exclude: ["onnxruntime-web"] },
    worker: { format: "iife" },
    build: { rollupOptions: { // probe.html is a dev-only diagnostic and pulls in the webgpu runtime; not built
      input: { main: "index.html", parity: "parity.html" } } },
  };
});
