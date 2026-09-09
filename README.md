# Benchmarking PPO, SAC and TD3 for Generalizable Robotic Arm Manipulation in MuJoCo

An **online** reinforcement-learning benchmark. PPO, SAC and TD3 each learn a pick-and-place task
on a simulated 4-DoF robotic arm in MuJoCo, from scratch, by interacting with the simulator and
generating all of their own experience.

> No dataset, no demonstrations, no offline buffer, no behaviour cloning. The simulator produces
> every transition the agents learn from.

Everything lives in one self-contained Colab notebook:
**[`Benchmarking_PPO_SAC_TD3_MuJoCo_Arm.ipynb`](Benchmarking_PPO_SAC_TD3_MuJoCo_Arm.ipynb)**.
Upload it to Colab, run all cells, done — no cloning, no downloads, no files written to disk except
results.

**Headline result:** under an identical task, reward, observation space, action space and 50k-step
interaction budget, **SAC solved the task (40% held-out test success, 92% grasp rate) while PPO and
TD3 reached 0%.** SAC was also the only algorithm to reach the 50% validation threshold at all,
doing so at ~39k environment steps.

---

## Contents

- [Research question](#research-question)
- [Environment](#environment)
- [Algorithms](#algorithms)
- [Experimental design](#experimental-design)
- [Results](#results)
- [Generalisation](#generalisation)
- [Robustness](#robustness)
- [Ablation](#ablation)
- [Visual demonstration](#visual-demonstration)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Future work](#future-work)

---

## Research question

> Under an identical task, reward, observation space, action space and environment-step budget, how
> do PPO, SAC and TD3 compare on a MuJoCo robotic-arm pick-and-place task in terms of **final
> success rate**, **sample efficiency**, **training stability**, and **generalisation to environment
> configurations never encountered during training**?

Deep RL papers routinely report that one algorithm beats another on manipulation, but the
comparisons are often confounded: different reward shaping, different observation preprocessing,
different interaction budgets, a single lucky seed, and evaluation on the very configurations the
agent trained on. This project controls all of that.

---

## Environment

### Robot and scene

A compact 4-DoF revolute arm — base yaw plus three pitch joints — on a pedestal in the middle of a
table, carrying a two-finger gripper. A 4.4 cm cube sits on the table and a translucent disc marks
the target. The MJCF model is written inline as a string, so nothing has to be downloaded and the
notebook cannot break because an asset URL moved.

![The scene](outputs/arm_benchmark/plots/scene_sample.png)

| | |
|---|---|
| Physics timestep | 4 ms |
| Frame skip | 5 (agent timestep = 20 ms) |
| Episode length | 200 steps = 4 s simulated |
| Arm DoF | 4 revolute + 2 mirrored finger slides |
| Reach | ≈ 0.59 m from the shoulder |

Arm links use MuJoCo's `gravcomp`, modelling the gravity-compensation loop every real industrial arm
ships with. Without it the position servos droop and all three algorithms spend their budget
learning to fight gravity rather than learning the task.

### Task

**Reach → grasp → transport → release → placement.** An episode succeeds when the cube has been
picked up, released, and comes to rest within 5 cm of the target.

Every episode randomises the *environment configuration*: initial joint angles (home pose ± 0.15
rad), cube position (radius 0.24–0.40 m, azimuth ±1.0 rad), cube yaw (full turn), cube mass
(0.04–0.08 kg) and target position (same polar ranges, ≥ 0.12 m from the cube).

### Grasping is abstracted — and why

When the agent commands *close* while the end-effector is within 5 cm of the cube centre, the cube
becomes kinematically attached. When it commands *open*, the cube is released with the
end-effector's current velocity and falls and settles under ordinary physics.

This is a deliberate simplification and the single most important design decision here.
Friction-based grasping with two thin fingers is extremely contact-sensitive; learning it from
scratch normally requires hindsight experience replay and millions of steps, which does not fit a
Colab session and would have produced a fragile project that fails for reasons unrelated to the
algorithms under comparison. The abstraction keeps every stage of the task — the agent still has to
reach the cube, decide *when* to close, carry it without dropping it, arrive over the target and
decide *when* to open — while removing contact-solver noise that says nothing about PPO vs SAC vs
TD3.

### Observation space

31-dimensional continuous vector:

| Index | Dim | Content |
|---|---|---|
| 0–3 | 4 | arm joint angles ÷ joint limits |
| 4–7 | 4 | arm joint velocities ÷ 8 rad s⁻¹ |
| 8 | 1 | gripper opening ∈ [0, 1] |
| 9–11 | 3 | end-effector position ÷ 0.5 m |
| 12–14 | 3 | end-effector linear velocity ÷ 2 m s⁻¹ |
| 15 | 1 | grasp flag |
| 16–18 | 3 | cube position ÷ 0.5 m |
| 19–21 | 3 | cube linear velocity ÷ 2 m s⁻¹ |
| 22–24 | 3 | target position ÷ 0.5 m |
| 25–27 | 3 | (cube − end-effector) ÷ 0.5 m |
| 28–30 | 3 | (target − cube) ÷ 0.5 m |

Normalisation uses **fixed analytic constants, not running statistics**. Wrapping PPO in
`VecNormalize` — the usual practice — would have given PPO a different observation space from SAC
and TD3 and broken the fairness of the benchmark.

### Action space

`Box(-1, 1, shape=(5,))`. `a[0:4]` are **incremental joint position targets**
(`target ← clip(target + 0.09·a, limits)`) tracked by MuJoCo PD servos; `a[4]` is the gripper
command (`> 0` closes).

**Why position control, not torques.** Torque control on a 4-DoF arm with a floating payload is
badly conditioned: the network must simultaneously produce gravity compensation, damping and task
motion, and PPO in particular struggles with it at Colab-scale budgets. Incremental position targets
give a smooth, bounded, well-scaled action space that suits an on-policy method and two off-policy
methods equally. The formulation is identical for all three, so it cannot favour any of them.

### Reward function

Dense and staged. Every component is configurable and several can be switched off for the reward
ablation.

| Component | Weight | Purpose |
|---|---|---|
| `−w_reach · ‖ee − cube‖` (only while not holding) | 1.0 | drives the approach; switches off after grasping so hovering is not rewarded |
| `−w_transport · ‖cube − target‖` | 1.2 | the actual task objective; active throughout |
| `+b_grasp` on the first grasp | 5.0 | makes the grasp discoverable without making it the whole objective |
| `+b_place` when released within 8 cm of the target | 10.0 | rewards *committing* to a release, which the transport term alone never does |
| `+b_success` on a completed placement | 25.0 | terminal bonus |
| `−p_object_lost` if the cube leaves the table | 10.0 | discourages flinging the cube |
| `−w_drop` if released > 16 cm from the target | 1.0 | discourages dropping en route |
| `−w_action · mean(a²)` | 0.05 | discourages bang-bang control |
| `−w_time` per step | 0.05 | rewards finishing quickly |
| `−w_collision` when an arm link touches the table or cube | 0.10 | discourages scraping the table |

**Measured scale**, from the notebook's own verification cell over 20 held-out configurations:

| Policy | Mean return | Success |
|---|---|---|
| Random | −140.6 ± 36.3 | 0/20 |
| Scripted inverse-kinematics controller | **+7.4 ± 13.1** | **20/20** (mean 85 steps) |

The scripted controller is a reference used only to prove the task is solvable and to establish the
reward scale. It produces no training data and no agent ever observes its behaviour.

---

## Algorithms

All three use the same `[256, 256]` MLP, the same learning rate (3 × 10⁻⁴), the same discount
(γ = 0.98) and the same environment-step budget.

**PPO** — on-policy. Collects a rollout with the current stochastic policy and takes several epochs
of minibatch gradient steps on a clipped surrogate objective. Discards its data after each update:
sample-inefficient but very stable. 8 parallel environments × 512 steps = 4096 transitions per
update, 10 epochs, clip 0.2, GAE λ = 0.95. The budget is rounded **down** to whole rollouts
(49,152 steps) so PPO never consumes more environment steps than the others.

**SAC** — off-policy, stochastic. Maximises reward plus policy entropy with an automatically tuned
temperature; twin Q-functions limit value overestimation. Replaying a buffer extracts far more from
each environment step than PPO, and the entropy term gives structured exploration.

**TD3** — off-policy, deterministic. Twin critics with a clipped double-Q target, policy updates
delayed to every second critic update, target-policy smoothing. Exploration comes from Gaussian
action noise (σ = 0.15) rather than an entropy term.

---

## Experimental design

### The split

There is no dataset here, so the split is over **environment configurations**:

| Split | How it is generated | Used for |
|---|---|---|
| **Training** | drawn fresh from an independent RNG stream on every reset — an agent essentially never revisits a configuration | online interaction and learning |
| **Validation** | fixed set of 30 configurations, seed 777 000 (20 at the `quick` preset) | checkpoint selection, algorithm selection, any tuning |
| **Test** | fixed set of 60 configurations, seed 999 000 (30 at `quick`), disjoint stream | the final number, read **once**, after the configuration is frozen |

The notebook verifies disjointness: the closest validation/test pair in the run below was 0.042
apart in 4-D pose space.

### Model selection

1. A callback evaluates on validation configurations every 5k steps and keeps the best checkpoint.
2. Each best checkpoint is then evaluated on the full validation set.
3. The winner is the algorithm with the highest mean validation success across seeds.
4. The configuration is **frozen** and written to `configs/frozen_configuration.json`, and only then
   is the held-out test set evaluated.

The notebook is ordered so this is visible: the freeze cell sits between selection and test.

### Fair-benchmarking checklist

| Held constant | How |
|---|---|
| environment | one env class, one MJCF model |
| observation space | fixed analytic normalisation inside the env — no `VecNormalize` for PPO only |
| action space | identical `Box(-1, 1, (5,))` |
| reward function | one shared config |
| task difficulty | identical randomisation ranges |
| interaction budget | identical; PPO rounded **down** to whole rollouts |
| network, learning rate, γ | `[256, 256]`, 3e-4, 0.98 everywhere |
| evaluation | the same function on the same fixed configuration sets |
| seeds | the same seed list for every algorithm |

---

## Results

**The run reported here used the `quick` preset: 50,000 environment steps per run, 2 seeds per
algorithm, 300,000 total environment steps.** Colab T4 GPU, Python 3.13, MuJoCo 3.13.0,
Gymnasium 1.3.0, Stable-Baselines3 2.9.0, PyTorch 2.11.0+cu128. Total training wall time ~34 min
(PPO 1.9, SAC 19.9, TD3 11.8), full notebook ~55 min.

### Final benchmark

| Algorithm | Val Success | Test Success | Mean Reward | Grasp Rate | Final Dist (m) | Steps to 50% Val | Stability (val σ) |
|:---|:---|:---|:---|---:|---:|:---|---:|
| **SAC** | **0.45 ± 0.21** | **0.40 ± 0.28** | **170.4 ± 23.0** | **0.92** | **0.107** | **39k** | 0.125 |
| TD3 | 0.05 ± 0.07 | 0.00 ± 0.00 | −102.7 ± 1.5 | 0.47 | 0.277 | not reached | 0.027 |
| PPO | 0.00 ± 0.00 | 0.00 ± 0.00 | −101.4 ± 2.3 | 0.10 | 0.309 | not reached | 0 |

± is the standard deviation across the 2 seeds. Mean reward, grasp rate and final distance are on the
held-out test set.

> **On the confidence intervals.** The notebook also emits a Student-t CI on the seed mean, which for
> SAC came out as `[-2.14, 2.94]` — a meaningless interval, because a t-interval from n = 2 has ~12.7
> degrees-of-freedom inflation and no clamping to [0, 1]. It is reported here as evidence of *how
> little* 2 seeds constrain the estimate, not as a result. The per-seed Wilson intervals over the 30
> test episodes are the informative numbers, and they are given below.

### Per-seed held-out test performance

| Algorithm | Seed | Test success | 95% Wilson CI (30 episodes) | Mean reward | Grasp rate |
|---|---:|---:|:---|---:|---:|
| SAC | 0 | 0.60 | [0.42, 0.75] | 186.6 | 0.97 |
| SAC | 1 | 0.20 | [0.10, 0.37] | 154.1 | 0.87 |
| TD3 | 0 | 0.00 | [0.00, 0.11] | −103.7 | 0.43 |
| TD3 | 1 | 0.00 | [0.00, 0.11] | −101.6 | 0.50 |
| PPO | 0 | 0.00 | [0.00, 0.11] | −103.1 | 0.20 |
| PPO | 1 | 0.00 | [0.00, 0.11] | −99.8 | 0.00 |

The seed-to-seed gap within SAC (0.60 vs 0.20) is larger than most reported differences between
algorithms in the literature, which is exactly why the benchmark uses multiple seeds.

Validation success (0.45) and test success (0.40) for SAC are close, so the policy is not
overfitting to the configurations used for selection.

![Validation vs test](outputs/arm_benchmark/plots/val_vs_test_success.png)

### Statistical comparison

Welch's t-test on the per-seed test success rates, SAC vs PPO: **t = 2.00, p = 0.295, Cohen's
d = 2.00**.

**No significance is claimed.** With 2 seeds per arm the test is badly underpowered, and p = 0.295 is
not evidence of equivalence either. The effect size is very large and the separation is
unambiguous in the raw numbers — SAC solves the task, the others never do — but establishing it
statistically would need the `standard` (3 seeds) or `full` (5 seeds) preset.

### Learning curves and sample efficiency

![Learning curves](outputs/arm_benchmark/plots/learning_curves.png)

All curves are plotted against **environment interaction steps**, never wall-clock time.

![Sample efficiency](outputs/arm_benchmark/plots/sample_efficiency.png)

| Algorithm | Steps to 50% validation success |
|---|---|
| SAC | **39k** (2/2 seeds; 38.3k and 40.0k) |
| PPO | never reached within 49,152 steps |
| TD3 | never reached within 50,000 steps |

The grasp-rate curve is the more informative view of the losing algorithms, since their success rate
never leaves zero:

![Secondary curves](outputs/arm_benchmark/plots/learning_curves_secondary.png)

**What the curves show.** SAC's validation success is flat at zero until ~30k steps, then rises
sharply to 0.62–0.75 by 40–45k — the task is discovered in a phase transition rather than gradually,
which is characteristic of staged reward structures where the grasp has to be found before the
transport reward becomes reachable. TD3 learned to grasp (0.43–0.50 test grasp rate) but essentially
never converted a grasp into a placement, suggesting its deterministic policy plus fixed Gaussian
noise did not explore the release decision. PPO barely learned to grasp at all (0.10 test grasp
rate) and its reward improved only slowly and monotonically from −127 to −87 — consistent with an
on-policy method that has simply not seen enough data at 49k steps.

**Wall-clock is not the same story as sample efficiency.** PPO finished its budget in 1.9 minutes
against SAC's 19.9, because 8 parallel environments and no replay make it ~10× faster per
environment step. On a wall-clock budget rather than a step budget the comparison would look
different — worth stating explicitly, since sample efficiency is the axis this benchmark measures.

### Behavioural metrics (held-out test set)

| Metric | PPO | SAC | TD3 |
|---|---:|---:|---:|
| Success rate | 0.000 ± 0.000 | **0.400 ± 0.283** | 0.000 ± 0.000 |
| Grasp rate | 0.100 ± 0.141 | **0.917 ± 0.071** | 0.467 ± 0.047 |
| Placement rate (released near target) | 0.000 ± 0.000 | **0.833 ± 0.189** | 0.000 ± 0.000 |
| Episode length (steps) | 198.6 ± 1.9 | **137.7 ± 39.8** | 200.0 ± 0.0 |
| Final cube–target distance (m) | 0.309 ± 0.016 | **0.107 ± 0.074** | 0.277 ± 0.002 |
| Collision rate | 0.076 ± 0.002 | 0.155 ± 0.143 | 0.159 ± 0.000 |
| Mean action magnitude | 0.103 ± 0.005 | 0.498 ± 0.082 | 0.314 ± 0.023 |
| Object lost | 0.017 ± 0.024 | 0.017 ± 0.024 | 0.000 ± 0.000 |

The gap between SAC's grasp rate (0.92) and success rate (0.40) locates the remaining failure mode
precisely: SAC almost always picks the cube up and usually releases it near the target (0.83), but
the release is often not accurate enough to land inside the 5 cm success radius. PPO and TD3 use up
the full 200-step episode every time; SAC terminates early on success, averaging 138 steps.

SAC's higher collision rate is not a defect — it is the only policy that actually descends to the
table surface to reach the cube.

---

## Generalisation

Every algorithm's policies were evaluated on five configuration distributions.

![Generalisation](outputs/arm_benchmark/plots/generalization.png)

| Suite | Shift | SAC success | Δ |
|---|---|---:|---:|
| `in_distribution` | none (reference) | 0.43 ± 0.32 | — |
| `heavy_object` | cube 0.14–0.24 kg (2–4× training) | 0.43 ± 0.32 | **+0.00** |
| `hard_initial_pose` | initial joint noise tripled to 0.45 rad | 0.40 ± 0.28 | −0.03 |
| `far_targets` | target radius 0.41–0.50 m, beyond training | 0.20 ± 0.21 | −0.23 |
| `unseen_sectors` | cube and target at azimuth 1.05–1.55 rad, never sampled | **0.00 ± 0.00** | **−0.43** |

PPO and TD3 scored 0.00 on every suite, so there is nothing to degrade.

**This is the most interesting result in the project.** SAC's policy is completely insensitive to
object mass (no degradation at 4× the training mass) and nearly insensitive to initial arm pose, but
it **collapses to zero** when the cube and target are moved into an azimuth sector it never trained
on. Distance generalises partially (0.20 at radii beyond training); *direction* does not generalise
at all.

The natural reading is that the policy learned a workspace-local mapping rather than a
configuration-invariant reaching strategy — it knows how to operate in the ±1.0 rad frontal sector it
saw and has no representation of the region outside it, even though the arm is mechanically
symmetric about the base yaw joint and the task there is no harder. Mass invariance comes cheaply
because gravity compensation and the kinematic grasp make mass nearly irrelevant to the dynamics the
policy sees.

This is a concrete argument for domain randomisation over the full azimuth range, and it is the kind
of failure that a benchmark evaluating only on the training distribution would never surface.

---

## Robustness

Four single-factor sweeps on the **selected** algorithm (SAC, both seeds, 25 held-out configurations
per point). Reported separately from the primary benchmark: these are stress tests, not the headline
result.

![Robustness](outputs/arm_benchmark/plots/robustness.png)

| Factor | Sweep | Success rate |
|---|---|---|
| Object mass (kg) | 0.03 → 0.06 → 0.10 → 0.16 → 0.24 | 0.50 → 0.43 → 0.27 → 0.50 → 0.30 |
| Target distance (m) | near 0.20–0.27 → mid → far → x-far 0.42–0.50 | 0.43 → 0.33 → 0.47 → **0.17** |
| Initial joint noise (rad) | 0.0 → 0.15 → 0.30 → 0.50 | 0.47 → 0.53 → 0.37 → 0.47 |
| Observation noise (σ) | 0.0 → 0.01 → 0.03 → 0.06 → 0.12 | 0.33 → 0.53 → 0.57 → 0.63 → 0.60 |

**Read these with caution.** Each point is 2 seeds × 25 episodes, and the seed-level standard
deviations (0.09–0.38) are as large as the trends. The mass and initial-pose sweeps are flat within
noise — consistent with the generalisation finding. The only clear signal is the drop to 0.17 at
extra-far targets (0.42–0.50 m), which agrees with the `far_targets` generalisation suite and is
partly a kinematic limit: 0.50 m is near the edge of the 0.59 m reach envelope.

The observation-noise sweep shows success *increasing* with noise (0.33 → 0.60). This is almost
certainly sampling noise rather than a real effect — the zero-noise baseline point happens to draw an
unlucky configuration set, and the differences sit inside the standard deviations. It is reported as
measured rather than quietly dropped, but no claim is made from it. A proper reading of this factor
needs more seeds.

---

## Ablation

The selected algorithm was retrained at a reduced budget (25,000 steps, 1 seed) with three
observation designs.

![Observation ablation](outputs/arm_benchmark/plots/ablation_observation.png)

| Observation mode | Dims | Validation success | Grasp rate | Mean reward |
|---|---:|---:|---:|---:|
| `joint_only` (proprioception only) | 9 | 0.00 | 0.00 | −109.9 |
| `joint_ee` (+ end-effector state) | 16 | 0.00 | 0.00 | −111.5 |
| `full` (task-aware) | 31 | 0.00 | **0.75** | **−62.9** |

**Inconclusive on the primary metric, informative on the secondary ones.** The 25,000-step ablation
budget sits *below* the ~30–40k point where SAC's success rate leaves zero in the main run, so all
three arms scored 0.00 success and the ablation cannot separate them on that measure. This is a
design flaw in the ablation budget, not a property of the observation modes, and it is stated rather
than glossed.

The grasp rate and reward do separate cleanly: the full observation reaches a 0.75 grasp rate and
−62.9 reward within 25k steps, while both reduced modes reach exactly 0.00 grasp rate and ~−110
reward — indistinguishable from an untrained policy. This is the expected result and confirms the
reduced modes behave as intended floors: without the cube and target positions the task is genuinely
unobservable, and no amount of training can solve it. Re-running the ablation at 60k+ steps (the
`standard` preset default) would give a success-rate comparison as well.

The reward ablation is available but disabled at the `quick` preset
(`cfg.run_reward_ablation = True` to enable).

---

## Visual demonstration

Training never renders a single frame. After the benchmark, the selected policy is rolled out again
on **held-out test** configurations with MuJoCo's offscreen renderer.

**Reference — the scripted IK controller solving the task** (`videos/reference_scripted.mp4`): this
is what a solved episode looks like, and the behaviour the agents have to discover on their own.

**Learned policy — SAC seed 0 on held-out test configurations**
(`videos/demo_SAC_seed0.mp4`, second angle in `videos/demo_SAC_front.mp4`):

| Episode | Success | Grasp | Return | Length | Final distance |
|---|:---:|:---:|---:|---:|---:|
| `test-0002` | ✓ | ✓ | 647.7 | 143 steps | 0.037 m |
| `test-0003` | ✓ | ✓ | 85.1 | 47 steps | 0.024 m |

Both show the full sequence: approach → grasp → transport → release → placement.

> GitHub does not play MP4 files inline in a README. The notebook also writes a `.gif` alongside each
> video; embed that instead, or link the MP4.

---

## Reproducibility

* Every tunable value lives in one configuration object. Nothing is hard-coded elsewhere.
* Python, NumPy and PyTorch seeds are set per run; training, validation and test configuration
  streams are seeded independently.
* The exact configuration that produced a result set is written to
  `configs/experiment_config.json`, the frozen selection to `configs/frozen_configuration.json`, and
  the validation/test configurations themselves to `configs/splits.json`.

### Running it

Open the notebook in Colab, set the runtime to GPU, run all cells. Section 1 installs dependencies
(and a software renderer if no GPU is present); section 4 selects the preset.

```python
PRESET = "quick"     # "smoke" | "quick" | "standard" | "full"
```

| Preset | Steps/run | Seeds | Measured / estimated total |
|---|---|---|---|
| `smoke` | 4 k | 1 | ~8 min — verifies every code path |
| `quick` | 50 k | 2 | **~55 min — the run reported above** |
| `standard` | 150 k | 3 | ~2–3 h — recommended for a reportable result |
| `full` | 400 k | 5 | ~8–12 h — strongest statistics |

### Output layout

The notebook creates this automatically:

```text
outputs/arm_benchmark/
├── models/         # SB3 .zip checkpoints, final and best-on-validation, per algorithm per seed
├── logs/           # per-environment Monitor CSVs
├── metrics/        # learning curves, per-episode evaluation records, all tables as CSV
├── plots/          # every figure in this README
├── videos/         # scripted reference + learned-policy demos (mp4 + gif)
├── configs/        # experiment config, frozen selection, the exact val/test splits
└── final_report/   # benchmark tables and summary.md as markdown
```

To reproduce this README's figures, commit `outputs/arm_benchmark/plots/` and
`outputs/arm_benchmark/videos/` alongside the notebook.

---

## Limitations

Stated plainly, because they bound what the results mean.

* **Two seeds.** The `quick` preset runs 2 seeds per algorithm. Confidence intervals on the seed mean
  are meaningless at n = 2 (SAC's came out as `[-2.14, 2.94]`), and Welch's test is badly
  underpowered (p = 0.295 despite a 0.40 vs 0.00 gap). The SAC-vs-PPO/TD3 separation is unambiguous
  in the raw numbers but is **not** statistically established here. Use `standard` or `full` for that.
* **Small interaction budget.** 50k steps is very small by manipulation-RL standards. Rankings at 50k
  need not hold at 5M. PPO especially is known to close ground with far more data, and its curve was
  still improving monotonically when the budget ran out — the honest reading of this benchmark is a
  **sample-efficiency** comparison, not an absolute verdict on the algorithms.
* **The ablation budget was too small.** At 25k steps all three observation modes scored 0.00 success,
  below the threshold where the task is learned at all. Only grasp rate and reward separate them.
* **Grasping is abstracted.** The kinematic attach removes the hardest part of real manipulation.
  Conclusions transfer to "which algorithm learns a multi-stage continuous-control task faster", not
  to "which algorithm grasps better".
* **Simulation only.** No hardware. MuJoCo contact and friction are approximations, and
  gravity-compensated position servos are more forgiving than a real controller.
* **Default hyper-parameters.** Literature-standard settings, not a per-algorithm sweep. A tuned PPO
  could beat an untuned SAC. The notebook includes a validation-only mini-sweep utility, disabled by
  default for compute reasons.
* **One task, one robot, low-dimensional state.** A 4-DoF arm, one pick-and-place task, ground-truth
  object and target poses rather than pixels. No claim about other morphologies or task families.
* **The robustness sweeps are noisy.** 2 seeds × 25 episodes per point, with seed-level σ comparable
  to the trends. Only the extra-far-target degradation is a clear signal.

---

## Future work

* **Domain randomisation over the full azimuth range** — the single highest-value next experiment,
  given that `unseen_sectors` was the one condition where the policy collapsed completely.
* **Longer budgets and more seeds** — re-run at `standard`/`full` to convert the observed separation
  into a statistically supported one, and to see whether PPO catches up.
* **Vision-based observations** — replace ground-truth poses with rendered camera images and a CNN
  encoder, which is where the sim-to-real gap actually lives.
* **Real contact-based grasping** — replace the kinematic attach with friction grasping, most likely
  combined with hindsight experience replay.
* **Sim-to-real transfer** — a system-identified model of a real low-cost arm, then physical
  evaluation of the same benchmark.
* **Harder manipulation** — stacking, insertion, sequential multi-object placement, obstacle
  avoidance.
* **More algorithms and goal conditioning** — DroQ/REDQ, TQC, and HER-style goal relabelling with a
  genuinely sparse reward, which would test whether the dense shaping here is doing more of the work
  than the algorithms are.

---

## License

MIT.
