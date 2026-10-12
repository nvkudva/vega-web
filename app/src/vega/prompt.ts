// Port of vega_common.serialize_state / option_texts / build_prompt.
// Token-for-token equality with the Python original is the contract here; every
// slice bound and template string below is load-bearing and verified against golden.json.
import type { InternalQ, QuestionDef, State } from "./types";

export interface Tok {
  encode(text: string): number[];
}

/** Python's json.dumps: separators default to ", " and ": ", unlike JSON.stringify. */
export function pyJsonDumps(v: unknown, ensureAscii: boolean): string {
  const esc = (s: string) => {
    let out = '"';
    for (const ch of s) {
      const c = ch.codePointAt(0)!;
      if (ch === '"') out += '\\"';
      else if (ch === "\\") out += "\\\\";
      else if (ch === "\n") out += "\\n";
      else if (ch === "\r") out += "\\r";
      else if (ch === "\t") out += "\\t";
      else if (c < 0x20) out += "\\u" + c.toString(16).padStart(4, "0");
      else if (ensureAscii && c > 0x7f) {
        if (c > 0xffff) {
          const u = c - 0x10000;
          out += "\\u" + (0xd800 + (u >> 10)).toString(16).padStart(4, "0");
          out += "\\u" + (0xdc00 + (u & 0x3ff)).toString(16).padStart(4, "0");
        } else out += "\\u" + c.toString(16).padStart(4, "0");
      } else out += ch;
    }
    return out + '"';
  };
  const go = (x: unknown): string => {
    if (x === null || x === undefined) return "null";
    if (typeof x === "string") return esc(x);
    if (typeof x === "boolean") return x ? "true" : "false";
    if (typeof x === "number") return String(x);
    if (Array.isArray(x)) return "[" + x.map(go).join(", ") + "]";
    return "{" + Object.entries(x as object).map(([k, v]) => esc(k) + ": " + go(v)).join(", ") + "}";
  };
  return go(v);
}

export function serializeState(state: State): string {
  return typeof state === "string" ? state : pyJsonDumps(state, false);
}

/** Request question -> internal form. Mirrors Vega._internal. */
export function toInternal(q: QuestionDef): InternalQ {
  let crit = q.criteria ?? null;
  if (q.type === "choice" && Array.isArray(crit)) {
    const d: Record<string, string | null> = {};
    for (const c of crit) d[String(c)] = null;
    crit = d;
  }
  // json.dumps at its default ensure_ascii=True here, unlike serialize_state
  const ins = typeof q.instructions === "string" ? q.instructions : pyJsonDumps(q.instructions, true);
  return { t: q.type, ins, crit };
}

const critText = (v: unknown) => (typeof v === "string" ? v : pyJsonDumps(v, false));

/** Option texts in label-index order. Boolean is always [no, yes], so p[1] is the probability of yes. */
export function optionTexts(q: InternalQ): string[] {
  if (q.t === "choice") {
    const crit = (q.crit ?? {}) as Record<string, unknown>;
    return Object.entries(crit).map(([k, v]) => (v === null || v === undefined || v === "" ? k : `${k} - ${critText(v)}`));
  }
  if (q.t === "score") return (q.crit as unknown[]).map((c, i) => `level ${i} - ${critText(c)}`);
  const crit = (q.crit ?? {}) as Record<string, unknown>;
  const no = crit["false"], yes = crit["true"];
  const set = (v: unknown) => v !== null && v !== undefined && v !== "";
  return [
    "no - " + (set(no) ? critText(no) : "the statement does not hold"),
    "yes - " + (set(yes) ? critText(yes) : "the statement holds"),
  ];
}

// Template 0 only: template 1 is the training-time invariance view (decide(views=3)).
const T = { head: "Observation:\n", q: "\n\nMeasurement ({t}): {ins}\nPossible outcomes:", opt: "\n* {o}", end: "\nOutcome:" };
const QUERY_FIRST = "Question to keep in mind while reading ({t}): {ins}\n\n";

export type Span = [number, number];
export interface Prompt {
  ids: number[];
  spans: { state: Span; question: Span; options: Span[] };
}

export interface PromptCfg {
  max_len: number;
  opt_max_tokens: number;
  query_first_max_state: number;
}

/** Vega's prompt, state first so its span never sees the question (causal attention), unless the
 *  state is short enough for the query-first preamble. The tokenizer adds no BOS for Qwen. */
export function buildPrompt(tok: Tok, stateText: string, q: InternalQ, cfg: PromptCfg, insMaxTokens = 256): Prompt {
  const maxLen = cfg.max_len;
  // replacer functions: a string replacement would expand "$&" and friends inside user text
  const fill = (s: string) => s.replace("{t}", () => q.t).replace("{ins}", () => String(q.ins));
  const opts = optionTexts(q);
  const head = tok.encode(T.head);
  let qids = tok.encode(fill(T.q)).slice(0, insMaxTokens);
  let oids = opts.map((o) => tok.encode(T.opt.replace("{o}", () => o)).slice(0, cfg.opt_max_tokens));
  const end = tok.encode(T.end);
  const sum = (xs: number[][]) => xs.reduce((a, x) => a + x.length, 0);
  let fixed = head.length + qids.length + sum(oids) + end.length;
  if (fixed > maxLen - 16) {
    // too many / too long options: shrink every option evenly, then the instructions
    const per = Math.max(3, Math.floor((maxLen - 16 - head.length - end.length - Math.min(qids.length, 64)) / Math.max(1, oids.length)));
    oids = oids.map((o) => o.slice(0, per));
    qids = qids.slice(0, Math.max(8, maxLen - 16 - head.length - end.length - sum(oids)));
    fixed = head.length + qids.length + sum(oids) + end.length;
  }
  let st = tok.encode(stateText);
  let pre: number[] = [];
  if (cfg.query_first_max_state && st.length < cfg.query_first_max_state) {
    pre = tok.encode(fill(QUERY_FIRST)).slice(0, Math.min(128, insMaxTokens));
    fixed += pre.length;
  }
  const room = Math.max(0, maxLen - fixed);
  if (st.length > room) {
    // the Python API truncates silently; a decision on a cut-off record is worse than none
    throw new Error(`state is ${st.length} tokens, only ${room} fit in max_len=${maxLen}`);
  }
  st = st.slice(0, room);
  const ids = [...pre, ...head];
  const s0 = ids.length;
  ids.push(...st);
  // an empty state still needs a span: fall back to the header tokens
  const state: Span = st.length ? [s0, ids.length] : [0, s0];
  const q0 = ids.length;
  ids.push(...qids);
  const question: Span = [q0, ids.length];
  const options: Span[] = [];
  for (const o of oids) {
    const a = ids.length;
    ids.push(...o);
    options.push([a, ids.length]);
  }
  ids.push(...end);
  return { ids, spans: { state, question, options } };
}
