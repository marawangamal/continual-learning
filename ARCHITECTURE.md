# Architecture: how CL methods are wired in this codebase

Four layers, one string (`importance_weighting`) threading through them. Every CL method in this codebase fits into this shape.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — CLI flags                                                    │
│  --ewc   --si   --owm   --xdg   --lwf   --agem   --replay=generative    │
│  ...                                                                    │
└────────────────┬────────────────────────────────────────────────────────┘
                 │
                 ▼  params/param_values.py:4      set_method_options(args)
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 2 — args (CLI → component booleans/strings)                      │
│                                                                         │
│  Each convenience flag expands into the *axes* it turns on:             │
│    --ewc  →  args.weight_penalty = True                                 │
│              args.importance_weighting = 'fisher'                       │
│              args.offline = True                                        │
│    --si   →  args.weight_penalty = True                                 │
│              args.importance_weighting = 'si'                           │
│    --lwf  →  args.replay = 'current'                                    │
│              args.distill = True                                        │
│    --agem →  args.replay = 'buffer'                                     │
│              args.use_replay = 'inequality'                             │
│    --icarl → args.prototypes = True; args.add_buffer = True; ...        │
│                                                                         │
│  Then set_default_values fills in λ, iters, lr, ... based on            │
│  experiment/scenario, and check_for_errors rejects bad combinations.    │
└────────────────┬────────────────────────────────────────────────────────┘
                 │
                 ▼  main.py:194-332            "CL-STRATEGY" sections
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 3 — model (args → model, one axis at a time)                     │
│                                                                         │
│  Each section is "if axis flag set, copy its args fields onto model":   │
│                                                                         │
│    l.198  CONTEXT-SPECIFIC COMPONENTS                                   │
│           └─ XdG: build model.mask_dict                                 │
│                                                                         │
│    l.217  PARAMETER REGULARIZATION                                      │
│           ├─ (a) fisher options (fisher_n, fisher_labels, offline, γ)   │
│           ├─ (b) weight_penalty axis (SI / EWC / KFAC / L2)   ← main    │
│           │       model.weight_penalty, model.importance_weighting,     │
│           │       model.reg_strength                                    │
│           └─ (c) precondition axis (OWM / NCL)                          │
│                                                                         │
│    l.258  FUNCTIONAL REGULARIZATION                                     │
│           ├─ distillation (LwF): model.replay_targets, model.KD_temp    │
│           └─ FROMP optimizer                                            │
│                                                                         │
│    l.275  REPLAY                                                        │
│           ├─ build generator (DGR / BI-R)                               │
│           ├─ model.replay_mode                                          │
│           └─ model.use_replay  (A-GEM)                                  │
│                                                                         │
│    l.313  MEMORY BUFFER                                                 │
│           └─ model.use_memory_buffer, budget, sample_selection, ...     │
└────────────────┬────────────────────────────────────────────────────────┘
                 │
                 ▼  training time — two branch sites
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 4 — branches                                                     │
│                                                                         │
│  Branch A: per-iteration loss term                                      │
│    models/classifier.py:289                                             │
│      if self.weight_penalty:                                            │
│          if importance_weighting == 'si':     self.surrogate_loss()     │
│          elif                       'fisher': self.ewc_loss() /         │
│                                               self.ewc_kfac_loss()      │
│          # elif                     'l2':     self.l2_loss()            │
│                                                                         │
│  Branch B: end-of-task state update                                     │
│    train/train_task_based.py:356                                        │
│      if importance_weighting == 'fisher' and (weight_penalty|precond):  │
│          model.estimate_fisher() / estimate_kfac_fisher()               │
│      if importance_weighting == 'owm'    and ...: estimate_owm_fisher() │
│      if importance_weighting == 'si'     and ...: update_omega(W, ε)    │
│      # if importance_weighting == 'l2' ...:       store_l2_anchor()     │
│                                                                         │
│  Branch C (SI-only): per-iteration state accumulation                   │
│    train_cl also calls update_importance_estimates(W, p_old) to grow W  │
│                                                                         │
│  Plus similar branch sites for replay, memory buffer, XdG masking —     │
│  all keyed off model fields that Layer 3 set.                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## The "axes" view

Every CL method turns on some combination of these orthogonal axes. Same axes, different combinations = different methods.

| axis                    | master flag             | selector / mode                           | main.py section |
|-------------------------|-------------------------|-------------------------------------------|-----------------|
| weight penalty (loss)   | `weight_penalty`        | `importance_weighting` ∈ {si, fisher, l2} | l.217 block (b) |
| precondition (grad)     | `precondition`          | `importance_weighting` ∈ {fisher, owm}    | l.217 block (c) |
| XdG masking             | `mask_dict is not None` | —                                         | l.198           |
| distillation            | `replay_targets=='soft'`| `KD_temp`                                 | l.258           |
| replay                  | `replay_mode`           | `'none\|current\|buffer\|generative\|all'`| l.275           |
| inequality replay       | `use_replay`            | `'normal\|inequality\|both'`              | l.302           |
| memory buffer           | `use_memory_buffer`     | `sample_selection`, `budget`              | l.313           |

Which axes each method uses:

```
method           weight_pen  precond  XdG  distill  replay       buffer
------           ----------  -------  ---  -------  ------       ------
EWC              ✓ (fisher)
SI               ✓ (si)
OWM                          ✓ (owm)
KFAC-EWC         ✓ (fisher,kfac)
NCL              ✓           ✓
XdG                                   ✓
LwF                                        ✓        current
ER                                                  buffer       ✓
A-GEM                                               buffer+ineq  ✓
DGR                                                 generative
BI-R                                       ✓        generative   (+distill, feedback, GMM)
FROMP            (via optimizer, not main penalty path)          ✓
iCaRL                                      ✓ (BCE)               ✓ (herding)
L2 (new)         ✓ (l2)
```

## Where everything lives (file index)

```
params/options.py               CLI → argparse (group by axis: add_cl_options, ...)
params/param_values.py
  ├─ set_method_options         CLI flag → axis booleans/strings (Layer 2)
  ├─ set_default_values         experiment/scenario → numeric defaults
  └─ check_for_errors           reject incompatible combinations
params/param_stamp.py           final args → filename for save/load

main.py                         Layer 3: args → model (the CL-STRATEGY sections)
main_task_free.py               same, for streaming setting
main_pretrain.py                single-context pretraining (no CL)

models/define_models.py         build Classifier / VAE / feature-extractor
models/classifier.py            train_a_batch — Branch A loss dispatch (l.289)
models/cl/continual_learner.py  the mixin — every weight_penalty method's
                                state and loss functions live here:
                                  SI:   register_starting_param_values,
                                        update_importance_estimates,
                                        update_omega, surrogate_loss
                                  EWC:  estimate_fisher, ewc_loss
                                  KFAC: estimate_kfac_fisher, ewc_kfac_loss
                                  OWM:  estimate_owm_fisher
                                  XdG:  apply_XdGmask, reset_XdGmask
models/cl/memory_buffer.py      replay buffer / iCaRL exemplars
models/cl/fromp_optimizer.py    FROMP custom optimizer
models/vae.py, cond_vae.py      DGR / BI-R generator
models/utils/loss_functions.py  distillation losses (LwF / BI-R)
models/utils/ncl.py             KFAC math helpers

train/train_task_based.py       train_cl — Branch B boundary dispatch (l.356),
                                  Branch C per-iter SI accumulation,
                                  plus replay sampling, buffer updates, etc.
train/train_stream.py           task-free counterpart
train/train_standard.py         pretraining loop

eval/callbacks.py               accuracy/loss/sample callbacks
eval/evaluate.py                per-context + aggregate accuracy
data/load.py                    get_context_set — split into tasks
data/{labelstream,datastream}.py  task-free streaming wrappers
```

## Adding a new weight-penalty method (L2 as worked example)

Five touchpoints, one per layer-boundary:

```
1. params/options.py:216
   └─ add CLI flag:    --l2

2. params/param_values.py:6  (set_method_options)
   └─ flag → axes:     args.weight_penalty = True
                       args.importance_weighting = 'l2'

3. models/cl/continual_learner.py  (near EWC block)
   └─ add the two method-specific functions:
       def store_l2_anchor(self): ...                ← Branch B target
       def l2_loss(self): ...                        ← Branch A target

4. models/classifier.py:289        (Branch A)
   └─ elif self.importance_weighting == 'l2':
          weight_penalty_loss = self.l2_loss()

5. train/train_task_based.py:356   (Branch B)
   └─ if model.importance_weighting == 'l2' and model.weight_penalty:
          model.store_l2_anchor()
```

No edits to `main.py` needed — block (b) at l.217 already copies `args.weight_penalty`, `args.importance_weighting`, and `args.reg_strength` onto the model, and that's all L2 needs.

Optional: extend `params/param_stamp.py` so L2 runs get a distinct filename from EWC runs (otherwise they write to the same `store/` paths).

### Sketch of the two new functions

```python
# in models/cl/continual_learner.py

def store_l2_anchor(self):
    """Snapshot current weights as L2 anchor θ*. Called at end of each context."""
    for gen_params in self.param_list:
        for n, p in gen_params():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.register_buffer(f'{n}_L2_anchor', p.detach().clone())

def l2_loss(self):
    """Σ (θ - θ*)²  summed over all regularized params. Returns 0 before first anchor."""
    try:
        losses = []
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    anchor = getattr(self, f'{n}_L2_anchor')
                    losses.append(((p - anchor) ** 2).sum())
        return 0.5 * sum(losses)
    except AttributeError:
        return torch.tensor(0., device=self._device())
```

### Test

```bash
python main.py --experiment=splitMNIST --scenario=task --l2 --lambda=100
```

Expected: better than the no-reg baseline, worse than `--ewc`. If your L2 matches EWC closely, something's wrong (likely `reg_strength` tuning or Fisher accidentally being stored).

## EWC as a concrete walkthrough

At the end of each task, measure **which weights mattered** for that task (= Fisher). Then when training the next task, penalize changes to those weights proportionally to how much they mattered. "Don't touch important weights; feel free to move the unimportant ones."

```
┌─── TASK 1 TRAINING ─────────────────────────────────────────┐
│ every iter:                                                 │
│   loss = CE(y, ŷ) + λ · ewc_loss()                          │
│                          └─ returns 0 (no Fisher yet)       │
└─────────────────────────────────────────────────────────────┘
                    ↓ task 1 finished
┌─── END-OF-TASK-1 ───────────────────────────────────────────┐
│ estimate_fisher():                                          │
│   for each param θᵢ:                                        │
│       Fᵢ = E[(∂log p(y|x,θ)/∂θᵢ)²]    ← squared gradients   │
│       θ*ᵢ = current value of θᵢ        ← "anchor"           │
│   register as buffers on the model                          │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─── TASK 2 TRAINING ─────────────────────────────────────────┐
│ every iter:                                                 │
│   loss = CE(y, ŷ) + λ · ½ · Σᵢ Fᵢ · (θᵢ - θ*ᵢ)²             │
│                        └── ewc_loss() now non-zero ─┘       │
└─────────────────────────────────────────────────────────────┘
```

**Snapshot step** — `estimate_fisher` (`models/cl/continual_learner.py:199-307`), called once per task boundary from `train/train_task_based.py:361`. It:

1. Loops over ~`fisher_n` samples of the just-finished task.
2. For each sample: forward pass → compute log-likelihood → backprop to get gradients.
3. **Squares the gradients** and accumulates them per-parameter (that's the diagonal Fisher).
4. Stores two buffers per parameter:
   - `<name>_EWC_prev_context` → θ* (the anchor, l.297)
   - `<name>_EWC_estimated_fisher` → F (the importance, l.302)

**Use step** — `ewc_loss` (`continual_learner.py:312-334`), called every iteration from `models/classifier.py:297`:

```python
losses.append( (fisher * (p - mean)**2).sum() )
...
return 0.5 * sum(losses)
```

That's just Σᵢ Fᵢ (θᵢ - θ*ᵢ)². Plain L2 is the same thing with Fᵢ = 1.

### Fisher-computation variants (ICLR blog post axis)

`estimate_fisher` has four ways to compute F (lines 231-285):

- `'all'` — weighted over *every* class label by softmax prob (the theoretically correct expected Fisher). O(classes) backward passes per sample.
- `'sample'` — sample one label from the predicted softmax (Monte Carlo).
- `'pred'` — use the argmax predicted label.
- `'true'` — use the ground-truth label → this is the **empirical Fisher**, not the true Fisher.

Paired with `fisher_n` (how many samples) and `fisher_batch` (how many per backward pass), this is the knob space the 2025 ICLR blog post studies. Same downstream `ewc_loss`, different Fisher estimates.

### Offline vs Online EWC

- **Offline EWC** (original Kirkpatrick et al.): keep a *separate* (F, θ*) pair per past task. `ewc_loss` sums over all of them (the outer loop over `context` at l.318).
- **Online EWC** (Schwarz et al.): maintain a *single* running F that accumulates with decay γ (l.301): `F ← γ·F_old + F_new`. Only one (F, θ*) stored. The `self.gamma` in `ewc_loss` (l.327) applies the decay at use-time.

Selected by `--offline` or (default) online; see the wiring in `main.py:226`.

## One-sentence summary

**CLI → args → model → branches, with `importance_weighting` as the method selector and a set of orthogonal axis booleans deciding what else gets turned on.** Every method you add is: one string value + the state/loss functions for that value + matching elif clauses at the two branch sites.
