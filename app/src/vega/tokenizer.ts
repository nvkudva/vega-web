import { PreTrainedTokenizer } from "@huggingface/transformers";
import type { Tok } from "./prompt";

/** transformers.js tokenizer built straight from tokenizer.json -- no model needed.
 *  Qwen adds no BOS, so plain encodes are exactly what build_prompt concatenates. */
export async function loadTokenizer(base: string): Promise<Tok> {
  const [tj, tc] = await Promise.all([
    fetch(`${base}/tokenizer.json`).then((r) => r.json()),
    fetch(`${base}/tokenizer_config.json`).then((r) => r.json()),
  ]);
  const t = new PreTrainedTokenizer(tj, tc);
  return { encode: (text) => t.encode(text, { add_special_tokens: false }) as number[] };
}
