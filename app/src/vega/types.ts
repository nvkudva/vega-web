export type QType = "choice" | "score" | "boolean";

/** Request shape, as vega_api.Vega.decide accepts it. */
export interface QuestionDef {
  type: QType;
  instructions: string | unknown;
  criteria?: Record<string, unknown> | unknown[] | null;
}
export type Questions = Record<string, QuestionDef>;
export type State = string | Record<string, unknown> | unknown[];

/** Internal form after Vega._internal: list criteria of a choice become {c: null}. */
export interface InternalQ {
  t: QType;
  ins: string;
  crit: Record<string, unknown> | unknown[] | null;
}

/** One calibration block: the base engine's or an adapter's. */
export interface Calibration {
  theta: number[][];
  conformal: Partial<Record<QType, Record<string, number>>>;
  abstain_threshold: Record<string, number>;
  unbound_threshold?: number;
  default_risk?: string;
}

/** The subset of vega_config.json (+ adapter jsons) the browser reads. */
export interface VegaConfig {
  max_len: number;
  opt_max_tokens: number;
  query_first_max_state: number;
  n_steps: number;
  memory_chunks: number;
  memory_min_tokens: number;
  memory_chunk_tokens: number;
  feature_layers: number[];
  option_layers: number[];
  calibration: Calibration;
  adapters: { name: string; gate_threshold: number; calibration: Calibration }[];
  vision_max_pixels: number;
  vision_merge_size: number;
}

export interface Answer {
  probs: Record<string, number>;
  confidence: number;
  prediction_set: (string | number | boolean)[] | null;
  abstain: boolean;
  unbound: boolean;
  settle_steps: number;
  energy: number;
  adapter?: string | null;
  gate?: number;
  choice?: string;
  score?: number;
  expected_score?: number;
  boolean?: number;
}

export interface VegaResponse {
  answers: Record<string, Answer>;
  latency_ms: number;
  n_tokens: number;
  views: number;
}

export const QTYPES: Record<QType, number> = { choice: 0, score: 1, boolean: 2 };
