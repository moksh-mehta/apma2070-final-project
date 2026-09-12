# Final Project — Learning Dynamical Systems from Data

By Moksh Mehta

APMA 2070 course project on the nonlinear pendulum:
compares **FNN**, **SympNet**, and **PINN** approaches for parameter
estimation and long-time trajectory prediction.

## Top-level deliverables
| File | Description |
| --- | --- |
| `Final_Report.docx` | 8-page report (Word) — open in Word or upload to Google Docs |
| `Final_Report.pdf`  | Same report as PDF for quick viewing |
| `Final_Presentation.pptx` | 12-slide deck (PowerPoint) |
| `Final_Presentation.pdf`  | Same deck as PDF for quick viewing |

## Folders
| Folder | Contents |
| --- | --- |
| `code/` | Six numbered Python scripts that reproduce every result and figure |
| `data/` | `train.txt`, `test.txt`, `full_trajectory.txt` from Störmer-Verlet |
| `figures/` | All 12 PNG figures used in the report and slides |
| `results/` | Saved model weights and per-step error arrays for each method |

## Reproducing from scratch

```bash
cd code
python 00_generate_data.py          # writes ../data/{train,test,full_trajectory}.txt
python 01_task0_sanity.py           # Hamiltonian + energy diagnostics
python 02_task1_fnn.py              # generic FNN baseline
python 03_task2_sympnet.py          # LA-SympNet with symplecticity test
python 04_task3_pinn.py             # PINN parameter estimation
python 05_task4_compare.py          # noise robustness comparison
```

Dependencies: `numpy`, `matplotlib`, `torch` (CPU is fine).

## Headline results

| Diagnostic | FNN | SympNet | PINN |
| --- | --- | --- | --- |
| Teacher-forcing MSE | 1.29e-4 | 5.85e-6 | n/a |
| Rollout MSE (99 steps) | 8.32e-2 | 6.68e-3 | 5.5e-6 |
| max \|det(D𝜑) − 1\| | 0.65 | 1.2e-7 | n/a |
| max \|H − H₀\| in rollout | 7.93e-2 | 1.46e-2 | 1.06e-3 |

**PINN recovers (m̂, l̂) = (1.0000, 0.9993)** — 0.07% from 40 training points.
Identifiability argument: with g fixed, both *m* and *l* are individually
identifiable from (θ, p_θ) data because the map (m, l) ↔ (α := ml², β := ml)
is a smooth bijection on the positive orthant.
