# vega on LocalLLaMA/typed-decisions (test split: 2050 decisions)

Modes (see the dataset card): **generalist** = never trained on these workflows, comparable to Jev; **specialist** = fitted on their train split, comparable to ModernBERT-base. Teacher self-agreement (0.735) is the ceiling; far above ~0.75 means the labeller's quirks were learned.

| model | mode | accuracy | soft acc | Brier | ECE | KL | TV | macro F1 | score MAE | within 1 | ms/case |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **vega, before fine-tuning** | generalist | 0.389 | 0.186 | 0.827 | 0.250 | 2.187 | 0.814 | 0.361 | - | - | 22 |
| **vega + typed-decisions adapter (gated), accuracy calibration** (default) | specialist | 0.763 | 0.680 | 0.314 | 0.026 | 0.652 | 0.320 | 0.798 | - | - | 22 |
| vega + typed-decisions adapter (gated), teacher calibration | specialist | 0.763 | 0.680 | 0.314 | 0.026 | 0.652 | 0.320 | 0.798 | - | - | - |
| Uniform | reference | 0.308 | 0.311 | - | - | - | - | - | - | - | - |
| Prior (label frequencies) | reference | 0.470 | 0.430 | - | - | - | - | - | - | - | - |
| MiniLM-L6 (22M) | specialist | 0.587 | 0.506 | - | - | - | - | - | - | - | - |
| ModernBERT-base (149M) | specialist | 0.646 | 0.542 | - | - | - | - | - | - | - | - |
| Perfect scenario understanding | ceiling | 0.704 | - | - | - | - | - | - | - | - | - |
| TypeSafe Jev 1.13.0 (card, 2026-09-18) | generalist | 0.727 | 0.580 | - | - | - | - | - | - | - | - |
| Teacher self-agreement | ceiling | 0.735 | - | - | - | - | - | - | - | - | - |

Paired (same 2050 decisions, McNemar): after vs before fine-tuning 864 / 97 (p=1.7e-154).
Two calibrations were fitted on the held-out train cases: against the correct answers (default; confidence tracks accuracy) and against the teacher's distributions (probabilities track the teacher's spread, lower KL, underconfident top label). Load the second with Vega(..., calibration="teacher").
Fine-tuning: best epoch 1, calibration-split soft NLL 1.8494 -> 0.6580, calibration mode thermal. Brier is summed over options against the gold distribution; the card may define it differently, so compare accuracy and soft accuracy with the card rows.

## Adapter (PEFT without forgetting)

The base engine is frozen and saved unchanged. One low-rank adapter (835937 trainable params, 5.6% of the engine) learned typed-decisions; a gate per question decides whether it applies (threshold 0.50). A closed gate returns the base engine's output exactly.

| items | mean gate | open | agrees with base engine | KL(base, adapted) |
|---|---|---|---|---|
| typed-decisions test | 1.000 | 100.0% | 43.4% | 3.9873 |
| typed-decisions calibration split | 1.000 | 100.0% | 45.7% | 3.9510 |
| vega replay, held out (old decision types) | 0.000 | 0.0% | 100.0% | 0.0000 |

| workflow | before | after |
|---|---|---|
| apibank | 0.512 | 0.895 |
| appliances | 0.538 | 0.851 |
| banking77 | 0.282 | 0.714 |
| clinc150 | 0.515 | 0.880 |
| esci | 0.216 | 0.564 |
| when2call | 0.423 | 0.904 |

| question type | before | after |
|---|---|---|
| choice | 0.389 | 0.763 |

| question | before | after |
|---|---|---|
| apibank/api | 0.512 | 0.895 |
| appliances/category | 0.538 | 0.851 |
| banking77/intent | 0.282 | 0.714 |
| clinc150/intent | 0.515 | 0.880 |
| esci/match | 0.216 | 0.564 |
| when2call/action | 0.423 | 0.904 |

