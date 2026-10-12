// Image as state: vega_vision.VegaVision's path. The image's tokens take the place of the state text
// in Vega's own prompt; the engine, adapters and calibration are unchanged. Zero-shot -- the engine was
// trained on text features -- and, as in vega_vision.py, the image path feeds the memory reader nothing.
import { Qwen2VLImageProcessor, RawImage } from "@huggingface/transformers";
import type { Span } from "./prompt";

/** The state text an image replaces: its single pad token is expanded to one per image feature. */
export const VISION_STATE = "<|vision_start|><|image_pad|><|vision_end|>";

export interface ImageInput {
  pixels: Float32Array; // [n_patches, 1536]
  grid: [number, number, number]; // (t, h, w) in patches
  nTokens: number; // image features after the 2x2 merge = pad tokens in the prompt
}

export interface VisionCfg {
  vision_max_pixels: number;
  vision_merge_size: number;
}

let processor: Qwen2VLImageProcessor | null = null;

/** Resize to the processor's pixel budget, normalise, cut into 2x16x16 patches (Qwen2-VL layout). */
export async function preprocess(base: string, cfg: VisionCfg, image: Blob): Promise<ImageInput> {
  if (!processor) {
    const pc = await fetch(`${base}/preprocessor_config.json`).then((r) => r.json());
    // VegaVision passes max_pixels=1024*32*32 to the processor, overriding the config's longest_edge.
    // The config omits do_normalize; Python defaults it to true, transformers.js to false, which
    // feeds the vision tower x/255 instead of (x/255 - 0.5) / 0.5.
    processor = new Qwen2VLImageProcessor({ do_normalize: true, ...pc, max_pixels: cfg.vision_max_pixels });
  }
  const raw = (await RawImage.fromBlob(image)).rgb();
  const { pixel_values, image_grid_thw } = await processor(raw);
  const g = Array.from(image_grid_thw.data as BigInt64Array, Number) as [number, number, number];
  const m = cfg.vision_merge_size;
  return { pixels: pixel_values.data as Float32Array, grid: g, nTokens: (g[0] * g[1] * g[2]) / (m * m) };
}

/** Replace the prompt's single image pad with nTokens pads, shifting every span after it. */
export function expandImage(ids: number[], spans: { state: Span; question: Span; options: Span[] }, padId: number, n: number) {
  const j = ids.indexOf(padId);
  if (j < 0) throw new Error("prompt has no image pad token");
  const out = [...ids.slice(0, j), ...new Array<number>(n).fill(padId), ...ids.slice(j + 1)];
  const sh = (a: number) => (a > j ? a + n - 1 : a);
  return {
    ids: out,
    spans: {
      state: [spans.state[0], sh(spans.state[1])] as Span,
      question: spans.question.map(sh) as Span,
      options: spans.options.map((o) => o.map(sh) as Span),
    },
  };
}

/** Qwen3.5 get_rope_index for one sequence with at most one image: text tokens count up on all three
 *  rows; the image block is its (t, h, w) grid offset by the current position, which then advances
 *  by max(h, w) of the merged grid. Returns [3 * L] row-major. */
export function ropeIndex(ids: number[], padId: number, grid: [number, number, number] | null, merge: number): BigInt64Array {
  const L = ids.length;
  const pos = new BigInt64Array(3 * L);
  let cur = 0;
  for (let i = 0; i < L; ) {
    if (grid && ids[i] === padId) {
      const [t, h, w] = [grid[0], grid[1] / merge, grid[2] / merge];
      let k = 0;
      for (let a = 0; a < t; a++)
        for (let b = 0; b < h; b++)
          for (let c = 0; c < w; c++, k++) {
            pos[i + k] = BigInt(cur + a);
            pos[L + i + k] = BigInt(cur + b);
            pos[2 * L + i + k] = BigInt(cur + c);
          }
      cur += Math.max(h, w);
      i += k;
    } else {
      pos[i] = pos[L + i] = pos[2 * L + i] = BigInt(cur);
      cur++;
      i++;
    }
  }
  return pos;
}
