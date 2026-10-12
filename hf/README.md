---
license: apache-2.0
library_name: onnx
pipeline_tag: text-classification
base_model: nandakishorm/vega-08b-public-intents
base_model_relation: quantized
tags:
  - vega
  - onnx
  - onnxruntime-web
  - quantized
  - int8
  - webassembly
  - in-browser
  - typed-decisions
  - calibration
  - conformal-prediction
  - image-classification
---

# Vega 0.8B, 8-bit, for the browser

**[▶ Try it — vega-onnx.pages.dev](https://vega-onnx.pages.dev)** — loads in a browser
tab, runs on your own machine, sends nothing anywhere.

This is [`nandakishorm/vega-08b-public-intents`](https://huggingface.co/nandakishorm/vega-08b-public-intents)
— a frozen Qwen3.5-0.8B with a small trained physics engine on top — as ONNX graphs that run on
onnxruntime-web. **840 MB for text**, plus 123 MB fetched only when you first use an image as the
state. Argmax agreement is unchanged on a 30-question text parity set (mean KL 4.0e-04) and on
the image parity set (mean KL 7.7e-04).

All credit for the model, the architecture, the training and the calibration belongs to
**[Nandakishor M](https://github.com/NandhaKishorM)** and **Convai Innovations**. The 8-bit
backbone is [onnx-community/Qwen3.5-0.8B-Text-ONNX](https://huggingface.co/onnx-community/Qwen3.5-0.8B-Text-ONNX).
This repository contributes only the graph cut, the engine export and the browser runtime.

| | |
|---|---|
| Base model | [nandakishorm/vega-08b-public-intents](https://huggingface.co/nandakishorm/vega-08b-public-intents) (0.8B, both adapters) |
| Backbone | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), frozen; 8-bit export by onnx-community |
| This conversion | [github.com/nvkudva/vega-web](https://github.com/nvkudva/vega-web) |
| Live demo | [vega-onnx.pages.dev](https://vega-onnx.pages.dev) |
| Licence | Apache 2.0, inherited from the base model and the backbone |

---

## What Vega is

Vega is **not a generative model**. A frozen language model reads a *state* and a *typed
question* once; a 14M-parameter engine places a particle in a 64-d decision space, puts one
Gaussian potential well under each option you listed, and integrates 12 damped steps. The
Boltzmann occupancy of the wells, at a temperature set by how settled the particle is, is
the answer. Every answer also carries a split-conformal prediction set, an abstain flag and
an out-of-distribution ("unbound") flag.

| Type | Answer | Use |
|---|---|---|
| `boolean` | `p(yes)` | Is this phishing? Should this escalate? |
| `choice` | one named option + the full distribution | Which queue? Which intent? |
| `score` | the expectation over ordered levels | How severe? How urgent? |

## Files

| File | Size | What |
|---|---|---|
| `v2/embed_q8.onnx` + `.data` | 294 MB | Qwen3.5-0.8B token embeddings, 8-bit |
| `v2/decoder_q8.onnx` + `.data` | 483 MB | decoder layers 0–19: `inputs_embeds` + 3-row M-RoPE `position_ids` in, `h13`, `h19` (residual stream after layers 13 and 19) out |
| `v2/engine.onnx` | 63 MB | the physics engine + `public_intents` and `typed_decisions` gated adapters, fp32 |
| `v2/vision_q8.onnx` + `.data` | 123 MB | Qwen3.5-0.8B vision encoder, 8-bit; only needed for images |
| `v2/vega_web.json` | 3 KB | prompt, pooling and vision settings, every calibration block (thermal θ, conformal, abstain, unbound) |
| `v2/preprocessor_config.json`, `v2/tokenizer.json`, `v2/tokenizer_config.json` | 13 MB | unchanged from Qwen3.5-0.8B |

`v1/` is the earlier text-only release (one 842 MB backbone graph, same accuracy). The path is
versioned on purpose: browser caches key on URL, so a re-export goes to a new version rather than
silently serving stale weights to anyone who already has the old one.

## How it is built

- **Backbone: a cut, not a re-export.** Vega's perception hooks decoder layers 13 and 19 and
  stops; the LM head and layers 20–23 never run. onnx-community's 8-bit multimodal export
  (MatMulNBits, block 32) already maps Qwen3.5's Gated DeltaNet layers to onnxruntime contrib
  ops, so its decoder is cut after layer 19. The cut follows tensors that `If` subgraphs read
  implicitly; this export keeps parts of each linear-attention layer inside them. The export fuses each
  residual add into the next layer's `SkipSimplifiedLayerNormalization`, so the two outputs
  are rebuilt as explicit `Add`s of the post-attention residual and the MLP output.
- **Engine: fp32, rewritten without data-dependent branches.** The memory reader, the score
  rail and early halting are each replaced by an exact unconditional equivalent (max |Δ|
  0.0 against the original on all 30 questions); the 12 steps are unrolled.
- **Pooling and calibration run in TypeScript**: span means, the final token, up to 16 chunk
  means of the state for the memory reader, then `τ = exp(b + w · [log1p kinetic, log1p
  nearest, steps/12, log k])` per question type and per route (base or adapter).

- **Image as state**, as in the base repo's `vega_vision.py`: the state text becomes
  `<|vision_start|><|image_pad|><|vision_end|>`, the pad is expanded to one token per merged 32×32
  patch (≤ 1024), the vision encoder's features replace those embeddings, positions are Qwen3.5's
  M-RoPE, and the engine gets no memory chunks. **Zero-shot**: the engine was trained on text
  features only.

## Accuracy

Against the PyTorch model (fp32, CPU), 30 questions spanning all three types, 2–14 options,
chunked and 6k-token states, dict states, non-Latin script and degenerate inputs:

| Backbone | Size | Argmax | Max \|Δp\| | Mean KL |
|---|---|---|---|---|
| fp32 cut (proves the cut) | 2745 MB | 30/30 | 0.0035 | 3.6e-06 |
| **8-bit (shipped)** | **842 MB** | **30/30** | **0.0503** | **4.0e-04** |
| 4-bit (rejected) | 497 MB | 27/30 | 0.4161 | 7.3e-02 |

In the browser the TypeScript port reproduces the 8-bit numbers exactly: token ids and spans
identical on 30/30, prediction sets 30/30, abstain 29/30. The one flip, and the max |Δp|,
are an empty state, where the reference confidence (0.645) sits near the abstain threshold.

Image path, against `vega_vision.py` on three synthetic documents (invoice, letter, receipt; 5
questions): the fp32 graphs match to KL 8e-07, proving the splice and positions; the 8-bit graphs
keep every answer at mean KL 5.9e-04 in Python and 7.7e-04 in the browser (the difference is the
browser's resize of the one image above the 1 MP budget). Real-document parity is still to do.

**4-bit is not usable**: the engine's wells and fitted temperatures sit on full-precision
hidden states, and 4-bit drift moves particles between basins.

## Running it

The full TypeScript port of the prompt, pooling and calibration lives in
[nvkudva/vega-web](https://github.com/nvkudva/vega-web) under `app/src/vega/`.

```ts
import * as ort from "onnxruntime-web/wasm";

const BASE = "https://huggingface.co/nvkudva/vega-onnx-q8/resolve/main/v2";
const load = (name: string) => ort.InferenceSession.create(`${BASE}/${name}.onnx`, {
  executionProviders: ["wasm"],
  externalData: [{ data: `${BASE}/${name}.onnx.data`, path: `${name}.onnx.data` }],
});
const [embed, decoder] = [await load("embed_q8"), await load("decoder_q8")];
const vision = await load("vision_q8");   // only for images
const engine = await ort.InferenceSession.create(`${BASE}/engine.onnx`, { executionProviders: ["wasm"] });
```

The decoder also takes zero-filled `past_conv.*`, `past_recurrent.*` and `past_key_values.*`
inputs (a first read). For text, `position_ids` is `0..L-1` on all three rows. For images, use
Qwen2-VL's image processor with `max_pixels = 1024·32·32` **and `do_normalize: true`** —
`preprocessor_config.json` omits it, and transformers.js defaults it to false. Prompt layout, which you must reproduce exactly:

```
[Question to keep in mind while reading (<t>): <ins>\n\n]   ← only when the state is < 4096 tokens
Observation:\n<state>\n\nMeasurement (<t>): <ins>\nPossible outcomes:\n* <option 0>\n* <option 1>…\nOutcome:
```

### Deployment notes

- **WebAssembly only**: onnxruntime-web's WebGPU `MatMulNBits` kernel does not take 8-bit.
- **Cross-origin isolation** for wasm threads: `Cross-Origin-Opener-Policy: same-origin` plus
  `Cross-Origin-Embedder-Policy: require-corp` (not `credentialless`, which Safari lacks).
- **Import `onnxruntime-web/wasm`**, not the default entry.

### Performance

Chromium, Apple silicon, 8 threads, one question:

| Prompt length | Latency |
|---|---|
| < 100 tokens | ~0.5–0.8 s |
| ~1.1k tokens | ~10 s |
| ~6k tokens | ~120 s |

The base model's 73,728-token context is real, but in the browser every question re-reads the
state, so long records are slow. A state that does not fit is refused, never truncated.

## Limits

These are properties of the base model; its [model card](https://huggingface.co/nandakishorm/vega-08b-public-intents)
documents them fully.

- Images are zero-shot (the engine never saw image features in training) and checked only on
  synthetic documents so far.
- `views=1` only (the API default); the 2- and 3-view test-time averaging is not exposed.
- The 4B checkpoint is not converted.

## Acknowledgements

[Nandakishor M](https://github.com/NandhaKishorM) and Convai Innovations built Vega and
released it under Apache 2.0. The Qwen team released Qwen3.5-0.8B; onnx-community exported it
to ONNX.

Graph cut, engine export and browser runtime by [nvkudva](https://huggingface.co/nvkudva).
