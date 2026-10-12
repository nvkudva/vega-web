// The bundler emits the asyncify and jsep wasm variants but not the plain threaded one
// the wasm-only build actually loads, so a production build hangs forever looking for a
// file that was never written. Serve ORT's runtime files ourselves and point env.wasm at them.
import { copyFile, mkdir, readdir } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";

const dist = dirname(createRequire(import.meta.url).resolve("onnxruntime-web"));
const out = new URL("../public/ort/", import.meta.url).pathname;
await mkdir(out, { recursive: true });
// Only the plain threaded runtime: src/vega/session.ts imports onnxruntime-web/wasm,
// which asks for this pair. Asking for a runtime that is not here surfaces as
// "no available backend found", which names no file -- so assert on the count.
const want = (f) => /^ort-wasm-simd-threaded\.(wasm|mjs)$/.test(f);
const files = (await readdir(dist)).filter(want);
if (files.length !== 2) throw new Error(`expected 2 ORT runtime files, found ${files}`);
for (const f of files) await copyFile(join(dist, f), join(out, f));
console.log(`copied ${files.join(", ")} -> public/ort/`);
