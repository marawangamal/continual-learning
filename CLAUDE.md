# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

PyTorch implementation of continual-learning experiments accompanying *Three types of incremental learning* (van de Ven et al., *Nature Machine Intelligence* 2022). The code supports both the **academic continual learning setting** (sharp, known context boundaries) and a **task-free** variant (fuzzy / unknown boundaries). The same code base is also used for the NeurIPS 2022 tutorial "Lifelong Learning Machines" (`NeurIPStutorial/`), the ICLR 2025 blog post on Fisher Information (`ICLRblogpost/`), and a stability-gap example (`StabilityGap/`).

See `ARCHITECTURE.md` for the deeper four-layer mental model of how CL methods are dispatched (CLI → args → model → branches) and a worked example of adding a new method.

## Environment

- Tested with Python 3.10.4, PyTorch 1.11.0, Torchvision 0.12.0. Dependencies in `requirements.txt`; `pyproject.toml` + `uv.lock` are the actual source of truth for this checkout.
- Run commands via `uv run ...` (no activation needed) or `source .venv/bin/activate` once per shell. `python main.py ...` works without `chmod` since the interpreter is invoked explicitly.
- `visdom` is intentionally excluded from the uv lockfile (its 0.2.4 sdist has a broken build). Only needed if you pass `--visdom` for live plots; safe to skip for every experiment in this repo.
- Entry scripts can also be invoked directly (`./main.py`) if you `chmod +x main*.py compare*.py all_results.sh` — they have a `#!/usr/bin/env python3` shebang.
- GPU autodetected: CUDA > Apple MPS > CPU. Use `--no-gpus` to force CPU.
- Outputs go under `./store/{datasets,models,plots,results}` (gitignored). `store/models/mM*` (saved models) is also gitignored.

## Common commands

Single experiment (academic setting):
```bash
python main.py --experiment=splitMNIST --scenario=task --si
```

Task-free (streaming) variant:
```bash
python main_task_free.py --experiment=splitMNIST --scenario=task --stream=fuzzy-boundaries
```

Pretrain convolutional feature extractor (only needed if a later CL run passes `--pre-convE`):
```bash
python main_pretrain.py --experiment=CIFAR10 --epochs=100 --augment --convE-stag=e100 --seed-to-stag --seed=1
```

Compare methods / sweep hyperparameters (produce summary PDFs):
```bash
python compare.py --experiment=splitMNIST --scenario=task
python compare_hyperParams.py --experiment=splitMNIST --scenario=task
python compare_replay.py --experiment=splitMNIST --scenario=task
python compare_task_free.py ...
python compare_hyperParams_task_free.py ...
```

Visdom live plots (optional, not installed by default): `pip install visdom`, run `python -m visdom.server`, then add `--visdom` to any run.

Print a param-stamp (used as filename for saved models/results) without training: add `--get-stamp`. Re-evaluate a previously saved model: add `--test` (disables `--train`).

Full reproduction pipeline for the article, tutorial, and blog post is scripted in `all_results.sh` (long; parallelize in practice).

There is no test suite or type-checker configured in this repo. Black formatting was applied in commit `5c6f312`.

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/): `<type>: <short description>`. Do **not** use parenthesized scopes in this repo (use `fix: ...`, not `fix(test): ...`).

Types used in this repo:
- `feat:` — new feature/capability
- `fix:` — bug fix
- `refactor:` — restructure without behavior change
- `perf:` — performance improvement
- `test:` — add/update tests
- `docs:` — documentation only
- `chore:` — tooling, ignores, cleanup, deps
- `revert:` — revert a previous commit

Keep the subject ≤ 72 chars, imperative mood ("add X", not "added X"), lowercase after the colon. Add a body only when the *why* isn't obvious from the diff. No Claude co-author trailers.

## Core options shared across scripts

- `--experiment`: `splitMNIST` | `permMNIST` | `CIFAR10` | `CIFAR100`
- `--scenario`: `task` | `domain` | `class` (task-IL, domain-IL, class-IL)
- `--contexts`: number of contexts (tasks)
- `--stream` (task-free only): `academic-setting` | `fuzzy-boundaries` | `random`
- Method "convenience" flags on `main.py`: `--si`, `--ewc`, `--lwf`, `--fromp`, `--xdg`, `--agem`, `--icarl`, `--brain-inspired`, `--gen-classifier`, `--separate-networks`, `--replay={generative,buffer,current,all,none}`, `--joint` (upper baseline), no-flag = "none" (lower baseline).

The method flags are *composable*: each one sets a bundle of underlying components (see `params/param_values.set_method_options`), and they can be combined or overridden with the lower-level component flags. Not every combination is tested.

## Architecture (one-minute tour)

### Top-level flow (`main.py`, `main_task_free.py`, `main_pretrain.py`)

All three follow the same shape: `handle_inputs() → run(args)`. `handle_inputs` builds an `argparse` parser by composing the option groups in `params/options.py` (`add_general_options`, `add_eval_options`, `add_problem_options`, `add_model_options`, `add_train_options`, `add_cl_options`), then runs `set_method_options` → `set_default_values` → `check_for_errors` in `params/param_values.py`. This three-stage pipeline is central: convenience flags expand into components, scenario/experiment-specific defaults are applied, and incompatible combinations are rejected before training begins.

`params/param_stamp.py` deterministically builds a long filename ("param stamp") from the final args; this stamp is used to save and reload models and results so multi-seed comparison scripts can look up prior runs.

### Data (`data/`)

- `data/load.py::get_context_set` returns `(train_datasets, test_datasets), config` for the chosen experiment/scenario. For the task-free entry point, the same context set is wrapped in a `DataStream` built from a `labelstream` (`SharpBoundaryStream`, `FuzzyBoundaryStream`, `RandomStream`).
- `data/manipulate.py` / `data/available.py` define dataset splits, permutations, and normalization.

### Models (`models/`)

- `models/define_models.py` is the single factory: `define_feature_extractor`, `define_classifier`, `define_vae`, plus `init_params` which handles loading pretrained conv layers.
- `models/classifier.py` (`Classifier`) composes `models/conv/nets.py` (ConvE) + `models/fc/nets.py` (fcE) + an output head. `classifier_stream.py` is the task-free counterpart. Its `train_a_batch` (l.289) is **Branch A** of the method dispatch — picks the per-iteration penalty loss based on `self.importance_weighting`.
- `models/feature_extractor.py` wraps just the frozen conv layers — used when `--freeze-convE` (and compatible) so data can be pre-featurized once up front.
- `models/cl/continual_learner.py` is the **central abstract mixin** (`ContinualLearner(nn.Module)`). Every CL classifier inherits from it. It owns the optimizer, scenario settings, and the bookkeeping for all regularization/replay methods: SI, EWC (including the Fisher-computation variants studied in the ICLR blog post), LwF / distillation, XdG masking, A-GEM, NCL/OWM. `self.param_list` controls which parameters are regularized (defaults to all `named_parameters`; OWM/KFAC restrict to `fcE` + `classifier`).
- `models/cl/memory_buffer.py` (and `_stream` variant) handles experience-replay buffers; `fromp_optimizer.py` implements FROMP's functional regularization.
- `models/cond_vae.py` / `vae.py` are the generative-replay models (Brain-Inspired Replay uses `cond_vae`).
- `models/generative_classifier.py` and `separate_classifiers.py` are alternative top-level models (multi-network methods).
- `models/utils/{loss_functions.py, modules.py, ncl.py}` hold shared losses, excitability modules, and Natural Continual Learning math (`additive_nearest_kf`).

### Training (`train/`)

Three parallel training loops, one per entry script:
- `train/train_task_based.py`: `train_cl` (main loop), `train_fromp`, `train_gen_classifier`. `train_cl` around l.356 is **Branch B** of the method dispatch — end-of-task state updates (Fisher estimation, SI omega update, OWM projection).
- `train/train_stream.py`: task-free counterparts (`train_on_stream`, `train_gen_classifier_on_stream`). Consolidation operations fire every `--update-every` iterations instead of at context boundaries.
- `train/train_standard.py`: single-context pretraining loop used by `main_pretrain.py`.

Training loops are callback-driven. `eval/callbacks.py` assembles callback lists (loss plots, accuracy eval, sample generation) that the loop invokes at configured iteration counts. `eval/evaluate.py` handles per-context and aggregate accuracy.

### Visualization (`visual/`)

- `visual/visual_plt.py`: matplotlib-based PDF summary generation (used with `--pdf`).
- `visual/visual_visdom.py`: live visdom plots (used with `--visdom`).

### Sub-project folders

`ICLRblogpost/`, `NeurIPStutorial/`, `StabilityGap/` each contain a single driver script (e.g. `compare_FI.py`, `compare_for_tutorial.py`, `stability_gap_example.py`) that reuses the core library. Their `README.md` files give the exact command lines reproducing the reported figures — check there first before re-deriving commands.

### Where to make changes

- **New CL method (weight-penalty family)**: see `ARCHITECTURE.md` for the full five-touchpoint recipe. Short version: add a CLI flag, map it to `importance_weighting='<name>'` in `set_method_options`, add the two method-specific functions on `ContinualLearner`, and add matching `elif` clauses at Branch A (`classifier.py:289`) and Branch B (`train_task_based.py:356`).
- **New dataset / scenario**: extend `data/available.py` + `data/load.py::get_context_set`, and add matching defaults in `params/param_values.set_default_values`.
- **New model architecture**: add to `models/conv/nets.py` or `models/fc/nets.py` and wire through `models/define_models.py`.
- **New comparison figure**: pattern off the existing `compare*.py` scripts; they call `main.run` / `main_task_free.run` in a loop over seeds & methods, reading results from the param-stamped cache.
