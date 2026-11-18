# Deep Exploration Homework — RND vs Bootstrapped DQN

Compare two deep-exploration agents — **Random Network Distillation (RND)** and
**Bootstrapped DQN with randomized priors** — on the **DeepSea** benchmark, on a
normal board and on a _noisy-TV_ variant.

## What you do

Open **`exploration_hw.ipynb`** in Google Colab (or Jupyter) and run it top to
bottom. You implement a handful of short functions across **three files**, one per
section, each edited through that section's own `%%writefile` cell and tested
before you move on:

1. **`hw/student_metric.py`** (Section B) — `coverage_entropy`, the exploration
   metric (coverage + visitation entropy).
2. **`hw/student_rnd.py`** (Section C) — the RND learning core: build its
   networks, the novelty bonus, the Double-DQN target, and the critic loss.
   (RND's ε-greedy action selection is provided.)
3. **`hw/student_bootstrap.py`** (Section D) — the Bootstrapped DQN core: build
   its ensemble, sample/act with a head, the evaluation (ensemble-mean) policy,
   the per-head target, and the masked loss.

Each function is short; the docstrings state what to return. Everything else — the
environment, agents, plotting, analysis, and artifact export — is provided in
`hw/` and you do not edit it.

## Running in Colab

1. Upload this whole folder (or unzip it) so the notebook sits next to `hw/`.
2. Run the cells in order. Section B is where you start filling in the functions;
   re-run the `%%writefile` cell and the `reload` cell after each edit.
3. Sections C and D quick-test your RND / Bootstrap code on a small DeepSea.
4. Section E runs the full normal-vs-noisy sweep (this takes a while). You may
   lower `TOTAL_STEPS` / `SEEDS` while exploring, but **submit a run that uses the
   given experiment settings** (`SIZE`, `SEEDS`, `TOTAL_STEPS`, `NOISE_DIM`,
   `NOISE_COLUMNS`); algorithm hyperparameters are yours to tune.
5. Section F shows the figures; Section G holds the analysis questions.

## Submitting

Set **`STUDENT_NUMBER`** at the top of the Setup section first. It is written into
`submission.zip` so the submission carries your student number.

The final cell writes **`submission.zip`** (< 10 MB): deterministic fingerprints
of your functions plus the run records from your sweep. **Submit
`submission.zip` together with your completed `exploration_hw.ipynb`.** Make sure
the sweep you submit uses the given experiment settings (see Section E).

## Local install (optional)

```bash
pip install -r requirements.txt
jupyter notebook exploration_hw.ipynb
```

## Files

```
exploration_hw.ipynb     the homework notebook (you work here)
requirements.txt         torch, numpy, matplotlib
hw/                       provided modules (do not edit, except the three
                          student_*.py files via the notebook's %%writefile cells)
```
