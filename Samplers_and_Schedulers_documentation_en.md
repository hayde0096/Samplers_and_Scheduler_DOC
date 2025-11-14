# Samplers and Schedulers: Overview and Grouping

This document organizes the common Stable Diffusion samplers and schedulers, emphasizing the idea of grouping similar content: learn the naming conventions, understand each family, then pair schedulers with usage scenarios for quick lookup and comparison.

## Document overview

- To understand what a sampler does, start with the basics of diffusion models.
  - During inference, a diffusion model starts from random noise and progressively “denoises” it into the final image.
  - This document only gives concise reminders instead of a full tutorial.
- Know how samplers and schedulers divide the work.
  - **Sampler**
    - A sampler is the collection of strategies a diffusion model uses during generation to reconstruct the image step by step.
    - It determines the computation at every step; different samplers trade off numerical stability, speed, detail, and stylistic consistency.
    - Put simply, the sampler decides *the path* the generation takes, affecting detail, sharpness, noise control, and overall style.
  - **Scheduler**
    - The scheduler decides how to arrange the **time steps** in the diffusion process—how denoising steps are allocated from noise to image.
    - Samplers decide *how* we move, schedulers decide *which* steps we take and how long each one is.
  - In short, **the scheduler sets the timetable and the sampler defines the motion**. Together they determine stability, speed, and quality.

## Sampler quick notes

- A sampler gradually converts noise into an image, influencing speed, detail, stability, and diversity.
- Low-order solvers (Euler, DPM Fast) are lightweight for rapid iterations; higher-order / SDE / Heun types (DPM++, Heun, LMS) capture more detail but run slower.
- As you increase steps, the scheduler (noise decay curve) matters more; good pairings can deliver finer texture or better stability at the same step count.

## Common suffixes

- `_ancestral`: enables ancestral sampling for more randomness and diversity.
- `_cfg_pp` / `cfg++`: implements CFG++ (see below) to fix off-manifold issues at low guidance scales.
- `_sde`: solver based on stochastic differential equations, usually with richer texture and randomness.
- `_gpu`: GPU-optimized or parallelized implementation.
- `_alt`: alternative numerical integration scheme.
- `_heun`: uses a Heun integrator to boost smoothness and suppress noise.
- `_2m` / `_3m`: higher-order multistep polynomial solvers; higher accuracy but slower.
- `seeds_2` / `seeds_3`: run multiple seeds in parallel or fuse their outputs.

## Sampler families at a glance

| Family / theme | Representative samplers | Key traits | Recommended scenarios |
| --- | --- | --- | --- |
| Euler | `euler`, `euler_ancestral`, `euler_cfg_pp`, `euler_dy` | Low order, fast, exploratory | Sketches, proofs of concept, low-step experiments |
| Heun | `heun`, `heunpp2` | Improved integration, smooth and stable, quite slow | Clean line art / noise suppression |
| DPM / DPM++ | `dpm_2/_a`, `dpmpp_2m/3m`, `dpmpp_sde`, `_heun`, `_cfg_pp` | High precision, photorealistic, rich variants | High-quality realism, SDXL, fine materials |
| LMS / PLMS | `lms`, `plms` | Photorealistic, stable, likes high steps | Photo feel, structural stability |
| DDIM / UniPC | `ddim`, `uni_pc`, `uni_pc_bh2` | Efficient at low steps, unified solver | Few steps with strong detail, benchmarking |
| DDPM | `ddpm` | Most stable but slowest | Research, ultra-stable pipelines |
| IPNDM | `ipndm`, `ipndm_v` | Iterative refinement | Extreme detail / precision |
| DEIS / Restart / Res Multistep | `deis`, `restart`, `res_multistep*` | Multi-step recovery, stackable with `_ancestral/_cfg_pp` | Enhanced-stability multi-step strategies |
| LCM | `lcm` | Latent-space control | Strong constraints, staged feature control |
| Gradient estimation | `gradient_estimation`, `*_cfg_pp` | Gradient corrections | Targeted optimization / research |
| Special / experimental | `er_sde`, `seeds_2/3`, `sa_solver*` | Custom or experimental solvers | Project-specific or stylized experiments |

## Family details

### Euler family (speed first)

- **Members**: `euler`, `euler_cfg_pp`, `euler_ancestral`, `euler_ancestral_cfg_pp`, `euler_dy`, `euler_smea_dy`, `euler_negative`, `euler_dy_negative`.
- **Traits**: Based on the simplest Euler integrator, so it is cheap and ideal for low-step rapid iterations. `_ancestral` adds randomness, `*_cfg_pp` brings CFG++ corrections, `euler_dy` / `smea_dy` emphasize dynamic step sizes or painterly smearing, and `*_negative` inject negative corrections to control style or stability.
- **Recommended**: Fast exploration, concept sketches, speed-sensitive workflows. Switch to DPM++, LMS, or Heun if you need more detail.

### Heun family (smooth & stable)

- **Members**: `heun`, `heunpp2`.
- **Traits**：Uses the improved Heun integrator for smoother transitions and less noise. `heunpp2` is a higher-order or engineered implementation, but plain `heun` is notably slow.
- **Recommended**: Clean edges, smooth transitions, and cases where speed is secondary; often paired with smoothing schedulers such as `smooth` or `beta`.

### DPM / DPM++ family (accuracy & realism)

- **Representative members**: `dpm_2`, `dpm_2_ancestral`, `dpm2_a`, `dpm_fast`, `dpm_adaptive`, `dpmpp_2s_ancestral`, `dpmpp_sde`, `dpmpp_sde_gpu`, `dpmpp_2m/_3m` plus `_alt`, `_cfg_pp`, `_sde`, `_heun`, `_gpu` variants.
- **Traits**: The mainstream high-quality sampler series. Different solver orders combined with SDE / Heun / CFG++ options let you balance speed, detail, and stability very precisely.
- **Recommended**: Photorealistic goals, large models (SDXL), and tasks demanding fine texture or consistent structure. When resources allow, favor `dpmpp_2m/3m` with Heun or SDE variants.

### LMS / PLMS (photorealistic bias)

- **Members**: `lms`, `plms`。
- **Traits**：Laplacian-pyramid style thinking with strong structural prior; high step counts yield clean, natural images.
- **Recommended**：Photo-grade realism or structural stability, especially with `karras` / `kl_optimal` schedulers.

### DDIM / UniPC / PLMS (quality & efficiency)

- **Members**: `ddim`, `uni_pc`, `uni_pc_bh2`, `plms` (if provided).
- **Traits**：DDIM balances speed and fidelity; UniPC is a newer unified solver that keeps detail with fewer steps.
- **Recommended**：Workflows with limited steps that still need detail; also common as baselines and for inversion tasks.

### DDPM / vanilla diffusion

- **Members**: `ddpm`。
- **Traits**：Original diffusion sampler—slowest but rock-solid stability.
- **Recommended**：Research, reproducing classic papers, workflows that demand extreme stability.

### IPNDM / IPNDM_V (iterative refinement)

- **Members**: `ipndm`, `ipndm_v`。
- **Traits**：Multi-stage iterative refinement; heavier compute but pushes detail further.
- **Recommended**：Extreme-detail materials or projects with stringent accuracy targets.

### DEIS / Restart / Res Multistep (multi-step recovery)

- **Members**：`deis`, `restart`, `res_multistep`, `res_multistep_cfg_pp`, `res_multistep_ancestral`, `res_multistep_ancestral_cfg_pp`。
- **Traits**：Use multi-step recovery, restarts, or differential tricks to boost stability; compatible with `_ancestral` and `_cfg_pp`。
- **Recommended**：Long sequences needing stability while re-injecting randomness mid/late trajectory。

### LCM (latent control)

- **Members**：`lcm`。
- **Traits**：Finer control over when latent-space features emerge。
- **Recommended**：Strongly constrained tasks or compositions needing mid-process intervention。

### Gradient / estimator samplers

- **Members**：`gradient_estimation`, `gradient_estimation_cfg_pp`。
- **Traits**：Explicitly leverage gradient estimates or corrective terms to steer generation。
- **Recommended**：Researchy or attribute-specific optimization work。

### Special / experimental samplers

- **Members**：`er_sde`, `seeds_2`, `seeds_3`, `sa_solver`, `sa_solver_pece` 等。
- **Traits**：Bespoke or experimental features such as multi-seed fusion (`seeds_2/3`), simulated annealing strategies (`sa_solver*`), or project-specific styles。
- **Recommended**：Proprietary aesthetics, experimental pipelines, or result-fusion workflows。

## Notes on CFG++ / `_cfg_pp`

- **Background**：Standard CFG reconstructs the posterior mean with the conditional noise term $\epsilon^c$, which drives the trajectory off the data manifold—especially at low guidance or during DDIM inversion。
- **Fix**：CFG++ swaps in unconditional noise $\epsilon^\varnothing$ when applying the Tweedie re-noising formula, yielding smoother trajectories that stick closer to the manifold。
- **Benefits**：
  - Greater stability at low guidance scales, avoiding mode collapse or saturation。
  - DDIM inversion / editing becomes more reversible with smaller error。
  - PF-ODE / generation trajectories stay straighter and easier to tune。
- **Practical tips**：
  - Names like `_cfg_pp`, `cfg++`, `cfgpp` usually indicate this variant; exact details depend on the project。
  - Pairing CFG++ with `ddim`, `dpmpp_*`, or `uni_pc` often delivers strong results even at low CFG (~2); raise guidance only if you need more forceful steering。
  - Distilled / accelerated models (SDXL Turbo, Lightning, etc.) benefit a lot。
- **Implementation notes**：Some projects add extra stability tricks (clip/scale). Check the relevant README or source for parameters。

## Sampler selection guide

- **Speed / sketching**：Prefer the `euler` family or `dpm_fast`。
- **High-quality realism**：`dpmpp_2m/3m`, `lms`, `plms`, `ddim` with `karras` / `kl_optimal`。
- **Diversity / experimentation**：Choose `_ancestral`, `_sde`, or `restart` variants。
- **Smooth & stable**：`heun`, `heunpp2`, `dpmpp_*_heun`。
- **Detail with few steps**：`ddim`, `uni_pc`, `plms`。

## Schedulers: overview & recommendations

### Role of a scheduler

Schedulers control how noise decays during sampling. Combined with a sampler, they change detail rendition, convergence speed, or randomness at a fixed step budget. Common options include：

### Basic / balanced schedules

- `simple`: Linear decay, ideal for quick tests and sketch phases; often paired with Euler or DPM Fast。
- `normal`: More balanced timetable for general-purpose runs or baselines。
- `linear_quadratic`: Linear early, quadratic later to emphasize different phases。

### Detail / high-frequency-friendly schedules

- `karras`: Allocates more step length near the end so low step counts still retain high-frequency detail; pairs well with `ddim`, `dpmpp_*`, `uni_pc`, `lms`。
- `kl_optimal`: Aims for minimal KL divergence（“optimal” decay）；first choice for realistic, fine-detail tasks。
- `beta` (and `beta_1_1`): Beta-curve schedules balancing detail and stability; `beta_1_1` is the uniform variant for controlled studies。

### Randomness / diversity-oriented schedules

- `sgm_uniform`: Equalizes per-step importance to balance randomness; good for steadily progressing generations。
- `ddim_uniform`: Uniform timetable within DDIM, useful for reproducibility or comparisons。
- `exponential`: Exponential decay that locks structure early; useful when speed is critical。

### Fine stage control

- `ays`, `ays+`, `ays_30`, `ays_30+`: Align Your Steps series that align critical steps; `30` variants target 30-step presets, `+` adds more knobs。
- `gits`: Generate In Transition Steps—injects key detail during transition steps; often paired with `ddim` / `euler` for crisp structure。

### Smooth / Heun partners

- `smooth`: Emphasizes gentle decay to suppress sudden noise at low steps; shines with Heun or `dpmpp_*_heun`。

## Scheduler quick reference

- **Low-step detail**：`karras`, `kl_optimal`, `beta`。
- **Speed vs. stability**：`simple`, `ddim_uniform`, `sgm_uniform`。
- **Stage control**：`ays*`, `gits`。
- **Maximize smoothness**：`smooth`, `beta` (especially with Heun-like integrators)。

## Scenario-based recommendations

| Scenario | Sampler | Scheduler | CFG starting point |
| --- | --- | --- | --- |
| Fast sketch / concept exploration | `euler` or `dpm_fast` | `simple` / `ddim_uniform` | 5.0~6.0 |
| High-quality realism (SD 1.5, high steps) | `dpmpp_2m` / `plms` / `ddim` | `karras` / `kl_optimal` | 7.0~8.0 |
| High-quality realism (SDXL with efficiency) | `dpmpp_3m_sde_heun` / `dpmpp_2m_sde_heun` | `karras` / `beta` | 5.5~7.0 |
| Illustration / stylized (line art + flats) | `ddim` / `plms` / `dpmpp_2m` | `karras` / `ays` / `gits` | 6.5~7.5 |
| Style experiments / diversity | `_ancestral` or `_sde` variants, `restart` | `sgm_uniform` / `beta` / `exponential` | 5.5~7.0 |
| Multi-LoRA stacks / general purpose | `*_cfg_pp` | `kl_optimal` / `simple` | 1~2 |

> CFG values depend on the model and prompt complexity. The table lists heuristic starting points—tweak by eye。
>
> Recently I mostly rely on CFGPP variants because I stack many LoRAs. They avoid burnt images even at low CFG (~2) while still producing solid results, but they pair poorly with the `karras` scheduler.
