# 采样器（Sampler）与调度器（Scheduler）的说明与分类

本文档整理了常见 Stable Diffusion 体系采样器（Sampler）与调度器（Scheduler），强调「相似内容归组」：先认识命名约定，再按家族了解特性，最后给出调度器与场景化组合，便于快速查找与对照。

## 文档综述

- 想要详细了解采样器的功能，需要先了解什么是扩散模型（diffusion model）。
  - 扩散模型在推理时会从随机噪声开始，逐步“去噪”生成最终图像。
  - 不过本文主要是进行一些简述，在此不做赘述。
- 了解采样器（Sampler）与调度器（Scheduler）。
  - 采样器（Sampler）
    - 采样器是扩散模型在生成阶段用于“如何一步步还原图像”的策略集合。
    - 采样器决定了每一步的具体计算方式。不同采样器在数值稳定性、速度、图像细节、风格一致性等方面各有不同的特性。
    - 简单来说，采样器决定**生成图像的“路径”**，不同的采样器会影响画面细节、锐度、噪声控制与整体风格。
  - 调度器（Scheduler）
    - 调度器用于 决定**扩散过程的时间步（time step）如何安排**，也就是从噪声到图像过程中如何分配去噪步骤。
    - 采样器负责“怎么走”，调度器负责“走哪几步以及步长多大”。
  - 简而言之，**调度器决定“时间表”，采样器决定“动作方式”**。两者配合决定扩散模型生成图像的稳定度、速度与质量。

## 采样器

- 采样器负责把噪声逐步还原为图像，影响速度、细节、稳定性、多样性。
- 低阶求解（Euler、DPM Fast）计算量小，适合快速迭代；高阶 / SDE / Heun 类（DPM++、Heun、LMS）细节更好但更慢。
- 随着步数提升，调度器（噪声衰减曲线）的影响愈明显；好的组合可以在同样步数下获得更细腻的纹理或更稳定的结构。

## 常见后缀

- `_ancestral`：引入 ancestral（祖先）采样，增加随机性与多样性。
- `_cfg_pp` / `cfg++`：实现 CFG++（见后文），修正 CFG 在低 guidance scale 下的 off-manifold 问题。
- `_sde`：基于随机微分方程（SDE）的求解，通常带来更丰富纹理与随机性。
- `_gpu`：针对于 GPU 的优化实现或并行化版本。
- `_alt`：替代数值积分方案。
- `_heun`：采用 Heun 积分器，提升平滑性与噪声抑制。
- `_2m` / `_3m`：更高阶的多步多项式求解器，精度高但更慢。
- `seeds_2` / `seeds_3`：多随机种子并行或结果融合。

## 采样器家族速览（按特性归组）

| 家族 / 主题 | 成员示例 | 关键词 | 推荐场景 |
| --- | --- | --- | --- |
| Euler | `euler`, `euler_ancestral`, `euler_cfg_pp`, `euler_dy` | 低阶、快速、探索 | 草图、概念验证、低步数实验 |
| Heun | `heun`, `heunpp2` | 改进积分、平滑、稳定，速度相当慢。 | 需要干净线条 / 噪声抑制的场景 |
| DPM / DPM++ | `dpm_2/_a`, `dpmpp_2m/3m`, `dpmpp_sde`, `_heun`, `_cfg_pp` | 高精度、写实、可调变体丰富 | 高质量写实、SDXL、精细材质 |
| LMS / PLMS | `lms`, `plms` | 写实、稳定、配合高步数 | 照片感、结构稳定需求 |
| DDIM / UniPC | `ddim`, `uni_pc`, `uni_pc_bh2` | 小步数效率、统一求解 | 少步数拿好细节、对比实验 |
| DDPM | `ddpm` | 最稳但最慢 | 研究、需要极高稳定性的流程 |
| IPNDM | `ipndm`, `ipndm_v` | 迭代精化 | 追求极致细节 / 精度 |
| DEIS / Restart / Res Multistep | `deis`, `restart`, `res_multistep*` | 多步恢复、可带 `_ancestral/_cfg_pp` | 增强稳定性的多步策略 |
| LCM | `lcm` | 潜在空间控制 | 强约束、逐步控制特征 |
| 梯度估计类 | `gradient_estimation`, `*_cfg_pp` | 梯度修正 | 定向优化或研究性用途 |
| 特殊 / 实验 | `er_sde`, `seeds_2/3`, `sa_solver*`| 自定义或实验求解器 | 项目内定制、风格化实验 |

## 家族详解（按组划分）

### Euler 家族（速度优先）

- **成员**：`euler`, `euler_cfg_pp`, `euler_ancestral`, `euler_ancestral_cfg_pp`, `euler_dy`, `euler_smea_dy`, `euler_negative`, `euler_dy_negative`。
- **特点**：基于最简单的 Euler 积分，计算成本低，适合低步数快速迭代。`_ancestral` 版本提供更多随机性，`*_cfg_pp` 结合 CFG++ 修正，`euler_dy`/`smea_dy` 主打动态步长或“涂抹”质感，`*_negative` 引入负向修正以控制风格或稳定性。
- **推荐**：快速探索、概念草图、速度敏感场景。如需更高细节可切换到 DPM++、LMS 或 Heun。

### Heun 家族（平滑与稳定）

- **成员**：`heun`, `heunpp2`。
- **特点**：使用 Heun 改进积分，转场更平滑、噪声更少。`heunpp2` 为更高阶或工程化实现。但是`heun`的速度相当之慢。
- **推荐**：需要干净边缘、平滑过渡以及不需要考虑速度的通用场景；也常与强调平滑的调度器（如 `smooth`, `beta`）搭配。

### DPM / DPM++ 家族（精度与真实感）

- **成员摘选**：`dpm_2`, `dpm_2_ancestral`, `dpm2_a`, `dpm_fast`, `dpm_adaptive`, `dpmpp_2s_ancestral`, `dpmpp_sde`, `dpmpp_sde_gpu`, `dpmpp_2m/_3m` 及其 `_alt`, `_cfg_pp`, `_sde`, `_heun`, `_gpu` 组合。
- **特点**：当前主流的高质量采样系列，可在速度、细节、稳定性之间通过不同求解器阶数与 SDE / Heun / CFG++ 组合进行细致权衡。
- **推荐**：写实、大模型（如 SDXL）、需要细腻纹理或高一致性的任务。在资源允许时优先尝试 `dpmpp_2m/3m` 的 Heun 或 SDE 版本。

### LMS / PLMS（写实取向）

- **成员**：`lms`, `plms`。
- **特点**：拉普拉斯金字塔思路，结构感强，配合高步数能得到干净自然的结果。
- **推荐**：追求照片级真实感或结构稳定性，特别适合与 `karras` / `kl_optimal` 调度器搭配。

### DDIM / UniPC / PLMS（高质量与效率并重）

- **成员**：`ddim`, `uni_pc`, `uni_pc_bh2`, `plms`（若实现中提供）。
- **特点**：DDIM 在速度与保真度间平衡；UniPC 是较新的统一方法，能在更少步数下保持细节。
- **推荐**：步数受限但希望保留细节的工作流；也是对比实验和反演任务的常用基线。

### DDPM / 原始扩散

- **成员**：`ddpm`。
- **特点**：最早的扩散采样方式，速度慢但稳定性最高。
- **推荐**：研究、复现经典论文、需要极高稳定性的流程。

### IPNDM / IPNDM_V（迭代精化）

- **成员**：`ipndm`, `ipndm_v`。
- **特点**：迭代多阶段精化，计算量较大但能进一步提升精细度。
- **推荐**：需要极致细节或针对特定材质、精度要求极高的项目。

### DEIS / Restart / Res Multistep（多步恢复）

- **成员**：`deis`, `restart`, `res_multistep`, `res_multistep_cfg_pp`, `res_multistep_ancestral`, `res_multistep_ancestral_cfg_pp`。
- **特点**：通过多步恢复、重启或差分策略增强稳定性，可与 `_ancestral`、`_cfg_pp` 等结合。
- **推荐**：想在较长序列中保持稳定、并在中后期重新注入随机性的流程。

### LCM（潜在控制）

- **成员**：`lcm`。
- **特点**：更细粒度地控制潜在空间的特征出现顺序。
- **推荐**：强约束任务或需要中途干预潜在特征的构图。

### 梯度 / 估计类采样器

- **成员**：`gradient_estimation`, `gradient_estimation_cfg_pp`。
- **特点**：显式利用梯度估计或修正项，改善生成方向感。
- **推荐**：研究性或需要针对性优化（如特定属性控制）的任务。

### 特殊 / 实验性采样器

- **成员**：`er_sde`, `seeds_2`, `seeds_3`, `sa_solver`, `sa_solver_pece`等。
- **特点**：通常是项目内定制或实验功能，如多种子融合（`seeds_2/3`）、模拟退火策略（`sa_solver*`）、特定风格预设。
- **推荐**：需要专有风格、实验性流程或结果融合的高级用例。

## CFG++ / `_cfg_pp` 说明

- **背景**：标准 CFG 在重构 posterior mean 时使用条件噪声 $\epsilon^c$，导致生成轨迹偏离数据流形（off-manifold），在低 guidance scale 或反演 (DDIM inversion) 场景尤其明显。
- **修正**：CFG++ 在 Tweedie 公式重噪时改用无条件噪声 $\epsilon^\varnothing$，使轨迹更平滑、更接近真实数据流形。
- **收益**：
  - 低 guidance scale 下更稳定，不易出现模式崩溃或饱和；
  - DDIM 反演 / 编辑更可逆，误差更小；
  - PF-ODE / 生成轨迹更直、更易调参。
- **实用建议**：
  - 名称如 `_cfg_pp`, `cfg++`, `cfgpp` 通常指代该实现；细节依项目而定。
  - 与 `ddim`, `dpmpp_*`, `uni_pc` 等采样器组合时，可在更低 CFG（1~2）获得相当好的结果；如需更强引导再逐步提高。
  - 蒸馏 / 加速模型（SDXL Turbo、Lightning 等）上收益尤为明显。
- **实现注意**：不同项目可能附带数值稳定技巧（clip/scale）。使用时请查阅对应实现的 README 或源码确认参数。

## 采样器选择指南

- **速度 / 草图**：优先 `euler` 系列、`dpm_fast`。
- **高质量写实**：`dpmpp_2m/3m`, `lms`, `plms`, `ddim`（配 `karras`/`kl_optimal`）。
- **多样性 / 实验**：选择带 `_ancestral`、`_sde` 或 `restart` 的变体。
- **平滑稳定**：`heun`, `heunpp2`, `dpmpp_*_heun`。
- **少步数还原细节**：`ddim`, `uni_pc`, `plms`。

## 调度器（Schedulers）说明与推荐

### 调度器作用

调度器决定噪声在采样过程中的衰减方式；与采样器搭配可在固定步数下改变细节表现、收敛速度或随机性。常见调度器如下：

### 基础 / 均衡调度

- `simple`：线性噪声衰减，适合快速测试与草图阶段；常配 Euler 或 DPM Fast。
- `normal`：比 `simple` 更平衡的时间表，用于通用场景或基准测试。
- `linear_quadratic`：前期线性、后期二次，便于强调不同阶段的权重。

### 细节 / 高频友好调度

- `karras`：对后期投入更多步长，低步数也能保留高频细节；与 `ddim`, `dpmpp_*`, `uni_pc`, `lms` 配合最佳。
- `kl_optimal`：以最小化 KL 散度为目标的“最优”噪声衰减；写实、精细任务首选。
- `beta`（及 `beta_1_1`）：基于 Beta 分布的可调曲线，兼顾细节与稳定；`beta_1_1` 为均匀版本，适合需要对比实验的流程。

### 随机性 / 多样性调度

- `sgm_uniform`：均匀化每步的重要性，平衡整体随机性；适合需要稳定渐进的生成。
- `ddim_uniform`：DDIM 框架下的均匀时间表，便于复现或对比实验。
- `exponential`：指数衰减，前期快速锁定结构；适合追求速度的场景。

### 精细阶段控制

- `ays`, `ays+`, `ays_30`, `ays_30+`：Align Your Steps 系列，通过对齐关键步骤来集中优化某些阶段；`30` 版本针对 30 步预设，`+` 增强控制选项。
- `gits`：Generate In Transition Steps—注入关键细节于过渡步骤；常与 `ddim` / `euler` 配合以获得清晰结构。

### 平滑 / Heun 搭档

- `smooth`：强调平滑衰减，减少小步数下的突变噪点；与 Heun、`dpmpp_*_heun` 协同效果好。

## 调度器选择速查

- **低步数保细节**：`karras`, `kl_optimal`, `beta`。
- **速度与稳定**：`simple`, `ddim_uniform`, `sgm_uniform`。
- **阶段控制**：`ays*`, `gits`。
- **追求平滑**：`smooth`, `beta`（尤其配 Heun / Heun-like 积分器）。

## 场景化推荐组合

| 场景 | 采样器 | 调度器| CFG 起点 |
| --- | --- | --- | --- |
| 快速草图 / 概念探索 | `euler` 或 `dpm_fast` | `simple` / `ddim_uniform` | 5.0–7.0 |
| 高质量写实（SD1.5, 高步） | `dpmpp_2m` / `plms` / `ddim` | `karras` / `kl_optimal` | 7.0–9.0 |
| 高质量写实（SDXL, 兼顾效率） | `dpmpp_3m_sde_heun` / `dpmpp_2m_sde_heun` | `karras` / `beta` | 5.5–7.0 |
| 插画 / 卡通（线稿 + 色块） | `ddim` / `plms` / `dpmpp_2m` | `karras` / `ays` / `gits` | 6.5–8.5 |
| 风格实验 / 多样性 | `_ancestral` 或 `_sde` 变体、`restart` | `sgm_uniform` / `beta` / `exponential` | 5.5–7.0 |
| 多lora叠加 / 较为通用 | `*_CFG_PP` | `kl_optimal` `simple` | 1~2 |

> CFG数值依模型与提示复杂度而异，上表给出常见启发式起点，可按视觉效果微调。
>
> 笔者最近使用的比较多的采样器基本都是各种采样器的CFGPP变体，因为叠加的lora相当之多，CFGPP变体采样器在低CFG（1~2）下避免了烧图的同时也同时能提供相当好的结果。但是与karras的搭配使用效果相当差。
