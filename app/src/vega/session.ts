// wasm-only entry: the default entry drags in the jsep runtime, whose .wasm is over
// Cloudflare Pages' 25MiB per-file cap. We only ever use the wasm EP.
import * as ort from "onnxruntime-web/wasm";
import { buildPrompt, serializeState, toInternal, type Prompt, type Tok } from "./prompt";
import { VISION_STATE, expandImage, preprocess, ropeIndex } from "./vision";
import { decideOne } from "./decide";
import { loadTokenizer } from "./tokenizer";
import { QTYPES, type Questions, type State, type VegaConfig, type VegaResponse } from "./types";

export interface LoadProgress {
  file: string;
  loaded: number;
  total: number;
  cached: boolean;
}

const CACHE = "vega-weights-v2";
// v1 held the text-only backbone, which v2 replaces with embed + decoder + vision
const STALE_CACHES = ["vega-weights-v1"];
const D = 1024;

/** Fetch with progress, backed by the Cache API so a reload does not re-download the weights. */
async function fetchCached(url: string, onProgress?: (p: LoadProgress) => void): Promise<Uint8Array> {
  const file = url.split("/").pop()!;
  let cache: Cache | undefined;
  try {
    cache = await caches.open(CACHE);
    const hit = await cache.match(url);
    if (hit) {
      const buf = new Uint8Array(await hit.arrayBuffer());
      onProgress?.({ file, loaded: buf.length, total: buf.length, cached: true });
      return buf;
    }
  } catch {
    // private mode / blocked storage: fall through and fetch without caching
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const total = Number(res.headers.get("content-length") ?? 0);
  const reader = res.body!.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress?.({ file, loaded, total, cached: false });
  }
  const buf = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  try { await cache?.put(url, new Response(buf)); } catch { /* quota */ }
  return buf;
}

/** Bytes the weight cache actually holds -- reported rather than assumed, since a
 *  quota failure on put() leaves a loaded session with nothing cached. */
export async function cachedWeightBytes(): Promise<number> {
  try {
    const cache = await caches.open(CACHE);
    let n = 0;
    for (const k of await cache.keys()) {
      const r = await cache.match(k);
      if (!r) continue;
      n += Number(r.headers.get("content-length") ?? 0) || (await r.blob()).size;
    }
    return n;
  } catch {
    return 0;
  }
}

export async function deleteWeightCache(): Promise<void> {
  try { await caches.delete(CACHE); } catch { /* blocked storage */ }
}

/** Zero recurrent / conv / KV state for every past input the cut decoder keeps: the first read
 *  of a sequence. Shapes come from the graph (batch 1, empty KV length). */
function emptyPast(sess: ort.InferenceSession): Record<string, ort.Tensor> {
  const feeds: Record<string, ort.Tensor> = {};
  for (const m of sess.inputMetadata) {
    if (!m.isTensor || !m.name.startsWith("past_")) continue;
    const shape = m.shape.map((d) => (typeof d === "number" ? d : d === "batch_size" ? 1 : 0));
    feeds[m.name] = new ort.Tensor("float32", new Float32Array(shape.reduce((a, b) => a * b, 1)), shape);
  }
  return feeds;
}

/** Perception._pooled_batch at batch 1: per feature layer the state-span mean, question-span
 *  mean and final token; per option the span mean at the option layer; and, for text states,
 *  the state span cut into up to `memory_chunks` equal pieces, mean-pooled at the deepest
 *  feature layer. Image states get no chunks, as in vega_vision.py. */
export function pool(cfg: VegaConfig, h13: Float32Array, h19: Float32Array, p: Prompt, withChunks = true) {
  const n = p.ids.length;
  const mean = (h: Float32Array, a: number, e: number, out: Float32Array, off: number) => {
    if (e <= a) return;
    for (let t = a; t < e; t++) for (let j = 0; j < D; j++) out[off + j] += h[t * D + j];
    for (let j = 0; j < D; j++) out[off + j] /= e - a;
  };
  const { state, question, options } = p.spans;
  const feats = new Float32Array(2 * 3 * D);
  [h13, h19].forEach((h, l) => {
    mean(h, state[0], state[1], feats, (l * 3 + 0) * D);
    mean(h, question[0], question[1], feats, (l * 3 + 1) * D);
    feats.set(h.subarray((n - 1) * D, n * D), (l * 3 + 2) * D);
  });
  const opts = new Float32Array(options.length * D);
  options.forEach(([a, e], k) => mean(h19, a, e, opts, k * D));
  const C = cfg.memory_chunks;
  const chunks = new Float32Array(C * D);
  const cmask = new Uint8Array(C);
  const m = state[1] - state[0];
  if (withChunks && m >= Math.max(cfg.memory_min_tokens, 2)) {
    const k = Math.max(2, Math.min(C, Math.floor(m / Math.max(1, cfg.memory_chunk_tokens))));
    // token i of the span goes to chunk floor(i * k / m), as torch.arange(n) * k // n
    let a = 0;
    for (let c = 0; c < k; c++) {
      let e = a;
      while (e < m && Math.floor((e * k) / m) === c) e++;
      mean(h19, state[0] + a, state[0] + e, chunks, c * D);
      a = e;
    }
    cmask.fill(1, 0, k);
  }
  return { feats, opts, chunks, cmask };
}

type Make = (name: string, external: boolean) => Promise<ort.InferenceSession>;

export class VegaSession {
  readonly cfg: VegaConfig;
  private base: string;
  private tok: Tok;
  private embed: ort.InferenceSession;
  private decoder: ort.InferenceSession;
  private engine: ort.InferenceSession;
  private past: Record<string, ort.Tensor>;
  private mk: Make;
  private vision: Promise<ort.InferenceSession> | null = null;
  private padId: number;

  private constructor(cfg: VegaConfig, base: string, tok: Tok, mk: Make, embed: ort.InferenceSession,
                      decoder: ort.InferenceSession, engine: ort.InferenceSession) {
    this.cfg = cfg;
    this.base = base;
    this.tok = tok;
    this.mk = mk;
    this.embed = embed;
    this.decoder = decoder;
    this.engine = engine;
    this.past = emptyPast(decoder);
    const pad = tok.encode("<|image_pad|>");
    if (pad.length !== 1) throw new Error(`<|image_pad|> is not a single token: ${pad}`);
    this.padId = pad[0];
  }

  static async load(base: string, onProgress?: (p: LoadProgress) => void, threads?: number): Promise<VegaSession> {
    // wasmPaths is not optional in production: left to the bundler, the build emits the
    // asyncify and jsep variants but not the plain threaded one, and session creation then
    // hangs with no error. In dev, Vite serves them from node_modules.
    if (import.meta.env.PROD) ort.env.wasm.wasmPaths = "/ort/";
    ort.env.wasm.numThreads = threads ?? Math.min(navigator.hardwareConcurrency || 4, 8);
    for (const name of STALE_CACHES) {
      try { await caches.delete(name); } catch { /* blocked storage */ }
    }
    const [cfg, tok] = await Promise.all([
      fetch(`${base}/vega_web.json`).then((r) => r.json() as Promise<VegaConfig>),
      loadTokenizer(base),
    ]);
    const mk: Make = async (name, external) => {
      const graph = await fetchCached(`${base}/${name}.onnx`, onProgress);
      const data = external ? await fetchCached(`${base}/${name}.onnx.data`, onProgress) : null;
      return ort.InferenceSession.create(graph, {
        executionProviders: ["wasm"],
        externalData: data ? [{ data, path: `${name}.onnx.data` }] : undefined,
      });
    };
    const embed = await mk("embed_q8", true);
    const decoder = await mk("decoder_q8", true);
    const engine = await mk("engine", false);
    return new VegaSession(cfg, base, tok, mk, embed, decoder, engine);
  }

  /** The vision encoder is only fetched the first time an image is decided. */
  loadVision(): Promise<ort.InferenceSession> {
    this.vision ??= this.mk("vision_q8", true);
    return this.vision;
  }

  /** questions: {id: {type, instructions, criteria}}. One decoder pass per question. With an
   *  image, the image is the state and `state` is ignored. */
  async decide(state: State, questions: Questions, image?: Blob): Promise<VegaResponse> {
    const t0 = performance.now();
    let img: { features: Float32Array; grid: [number, number, number]; n: number } | null = null;
    if (image) {
      const pre = await preprocess(this.base, this.cfg, image);
      const vision = await this.loadVision();
      const { image_features } = await vision.run({
        pixel_values: new ort.Tensor("float32", pre.pixels, [pre.pixels.length / 1536, 1536]),
        image_grid_thw: new ort.Tensor("int64", BigInt64Array.from(pre.grid, BigInt), [1, 3]),
      });
      img = { features: image_features.data as Float32Array, grid: pre.grid, n: pre.nTokens };
    }
    const text = img ? VISION_STATE : serializeState(state);
    const out: VegaResponse = { answers: {}, latency_ms: 0, n_tokens: 0, views: 1 };
    for (const [qid, qdef] of Object.entries(questions)) {
      const q = toInternal(qdef);
      let p = buildPrompt(this.tok, text, q, this.cfg);
      if (img) p = expandImage(p.ids, p.spans, this.padId, img.n);
      const L = p.ids.length;
      const { inputs_embeds } = await this.embed.run({
        input_ids: new ort.Tensor("int64", BigInt64Array.from(p.ids, BigInt), [1, L]),
      });
      const emb = inputs_embeds.data as Float32Array;
      if (img) {
        // image features replace the pad tokens' embeddings, in order
        let f = 0;
        for (let t = 0; t < L; t++) if (p.ids[t] === this.padId) emb.set(img.features.subarray(f * D, ++f * D), t * D);
      }
      const { h13, h19 } = await this.decoder.run({
        ...this.past,
        inputs_embeds: new ort.Tensor("float32", emb, [1, L, D]),
        attention_mask: new ort.Tensor("int64", new BigInt64Array(L).fill(1n), [1, L]),
        position_ids: new ort.Tensor("int64", ropeIndex(p.ids, this.padId, img?.grid ?? null, this.cfg.vision_merge_size), [3, 1, L]),
      });
      const f = pool(this.cfg, h13.data as Float32Array, h19.data as Float32Array, p, !img);
      const k = p.spans.options.length;
      const C = this.cfg.memory_chunks;
      const r = await this.engine.run({
        feats: new ort.Tensor("float32", f.feats, [1, 2, 3, D]),
        opts: new ort.Tensor("float32", f.opts, [1, k, 1, D]),
        opt_mask: new ort.Tensor("bool", new Uint8Array(k).fill(1), [1, k]),
        qtype: new ort.Tensor("int64", BigInt64Array.from([BigInt(QTYPES[q.t])]), [1]),
        chunks: new ort.Tensor("float32", f.chunks, [1, C, D]),
        chunk_mask: new ort.Tensor("bool", f.cmask, [1, C]),
      });
      out.answers[qid] = decideOne(this.cfg, q, {
        logits: Array.from(r.logits.data as Float32Array).slice(0, k),
        kinetic: (r.kinetic.data as Float32Array)[0],
        nearest: (r.nearest.data as Float32Array)[0],
        steps: Number((r.steps_used.data as BigInt64Array)[0]),
        gates: Array.from(r.gates.data as Float32Array),
      });
      out.n_tokens += L;
    }
    out.latency_ms = Math.round(performance.now() - t0);
    return out;
  }
}
