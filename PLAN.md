# vega-web — Vega 0.8B in the browser

## What Vega is

Not a generative LLM. A typed decision model: a **frozen Qwen3.5-0.8B** reads a prompt once,
and a small trained **physics engine** (14.3M params, plus gated adapters) turns pooled hidden
states into a calibrated distribution over the question's options.

```
prompt ids ──> Qwen3.5-0.8B layers 0..19 (frozen) ──> h13, h19 [L, 1024]
  pool: per layer {13,19}: state-span mean, question-span mean, final token     -> feats [2,3,1024]
        per option: option-span mean at layer 19                                 -> opts  [K,1,1024]
        state span cut into up to 16 equal chunks, mean at layer 19 (>=128 tok)  -> chunks [16,1024]
  engine: world latent (+ memory reader over chunks) -> particle z0, p0
          one Gaussian well per option (score options on a 1-d rail)
          12 damped symplectic steps with a selective-SSM thermostat
          logits = -well energy at the final position
  calibration: tau = exp(b + w . [log1p kinetic, log1p nearest, steps/12, log k])
               per qtype, per route (base engine or the adapter whose gate >= 0.5)
               -> softmax(logits / tau), LAC conformal set, abstain, unbound
```

Prompt (`build_prompt`, template 0): `[query-first preamble if state < 4096 tok] "Observation:\n"
<state> "\n\nMeasurement (<t>): <ins>\nPossible outcomes:" ("\n* <option>")* "\nOutcome:"`.
Qwen has no BOS. Question types: `choice`, `score`, `boolean` (`[no, yes]`, answer `p[1]`).

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Checkpoint | `nandakishorm/vega-08b-public-intents` root (0.8B), both adapters (`public_intents`, `typed_decisions`), `adapters="auto"` | the default `Vega(model_dir)` configuration; parity is defined against it |
| Backbone source | **onnx-community/Qwen3.5-0.8B-Text-ONNX**, graph-cut | Qwen3.5's Gated DeltaNet layers need contrib ops (`LinearAttention`, `CausalConvWithState`, `GroupQueryAttention`) that onnx-community's exporter already emits; re-exporting from torch would mean re-solving that |
| Graph cut | outputs `h13`, `h19` = explicit `Add(post_attention residual, mlp down_proj)`; everything after layer 19 and the LM head dropped | Perception hooks layers 13 and 19 and raises `StopForward` after 19; logits are never read. The export fuses each residual add into the next layer's `SkipSimplifiedLayerNormalization`, so the sums are rebuilt explicitly |
| Backbone precision | **onnx-community `model_quantized`: 8-bit MatMulNBits + 8-bit GatherBlockQuantized**, 842 MB | measured, see table below; q4 breaks calibration |
| Engine precision | **fp32**, 63 MB, one file | 63 MB is small next to the backbone, and the 12-step settle plus the fitted thermal temperatures sit directly on its outputs |
| Engine export | unconditional rewrite (`export_engine.py::ExportEngine`) | memory-reader / score-rail / halt branches are data-dependent; each is replaced by an exact equivalent (max \|d\| 0.0 vs the original on all 30 golden questions) |
| Pooling | TypeScript, not ONNX | span means and chunking are cheap, index-heavy and easy to port; keeps the backbone graph a pure cut of onnx-community's |
| Batch | 1 per question, full read each time | measured equal to batched `decide()` (max \|dp\| 3.7e-07). Prefix KV sharing for long states is an optimisation, not a correctness need |
| Runtime | onnxruntime-web **wasm**, threaded, main thread | inherited from laya-web: ORT-web WebGPU has no 8-bit MatMulNBits; threaded wasm hangs in workers in prod builds |
| Truncation | refuse (throw) a state that does not fit `max_len` | the model card's "no silent truncation"; `build_prompt` itself would cut it |
| Tokenizer | `Qwen/Qwen3.5-0.8B` `tokenizer.json` via transformers.js | 30/30 token ids and spans identical to Python |
| App | laya-web's Vite + React + TS + bun scaffold, COOP/COEP `require-corp` | same hosting constraints |

## Measured (Python ORT CPU vs golden, 30 questions, B=1)

| Backbone | data | argmax | max \|dp\| | mean KL |
|---|---|---|---|---|
| fp32 cut | 2745 MB | 30/30 | 0.0035 | 3.6e-06 |
| **q8 cut** | **842 MB** | **30/30** | **0.0503** (state_empty) | **4.0e-04** |
| q4 cut | 497 MB | 27/30 | 0.4161 | 7.3e-02 |

The fp32 row proves the graph cut; the q8 and q4 rows are quantization alone.

## Rejected

- **q4 / q4f16 backbone** — 3 argmax flips and KL 0.07; the engine's wells and thermal
  temperatures were fitted on full-precision hidden states, and 4-bit drift (d_feat ~5% relative)
  moves particles between basins.
- **fp16 backbone** — ~1.5 GB, and no WebGPU path to make it pay (see runtime).
- **Own torch.onnx export of Qwen3.5** — Gated DeltaNet in torch falls back to a reference
  chunked loop; onnx-community already maps it to ORT contrib ops.
- **Vision encoder** — text only for v1; `vega_vision.py` is a separate path.
- **views=2/3 test-time averaging** — doubles/triples latency; views=1 is the API default.
- **4B checkpoint** — Qwen3.5-4B backbone, several GB even at 8-bit.

## Constraints

- Context: `max_len` 73,728, but wasm prefill is the limit in practice; long states are slow
  (every question re-reads the state).
- Engine and calibration must stay fp32.
- Boolean probs keys are `"False"`/`"True"` (Python `str(bool)`); keep for response parity.
- The cut graph keeps the `past_*` inputs; the browser feeds zero state (first read).

## Revisions
- Hosting: Cloudflare Pages project **vega-onnx** (vega-onnx.pages.dev). vega-web.pages.dev is owned by an unrelated site; the repo keeps its vega-web name.
- **Image as state (v2 weights).** The text-only backbone is replaced by onnx-community's multimodal
  export: `embed_q8` (294 MB) + `decoder_q8` (layers 0-19, takes `inputs_embeds` + 3-row M-RoPE
  `position_ids`, 483 MB) + `vision_q8` (123 MB, fetched on first image). Text total drops 905 -> 840 MB.
  Image path = `vega_vision.py`: `<|vision_start|><|image_pad|><|vision_end|>` as the state text, the pad
  expanded to one token per merged patch, image features spliced over those embeddings, no memory
  chunks. Zero-shot: the engine was trained on text features. `cut_backbone.py` now follows tensors
  read implicitly by `If` subgraphs (this export keeps parts of each linear-attention layer in them;
  the first cut silently dropped them). transformers.js's Qwen2VL processor needs `do_normalize: true`
  passed explicitly: the config omits it, Python defaults true, JS defaults false.
  Measured: text q8 unchanged (argmax 30/30, KL 3.99e-4, browser identical); image fp32 KL 8e-7 (proves
  splice + rope), image q8 Python KL 5.9e-4, browser KL 7.7e-4, argmax 5/5 (3 synthetic documents; the
  extra browser drift is the resize of the one image over max_pixels). ~4-10 s per image question.
