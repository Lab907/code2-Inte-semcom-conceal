# RF Fixed Pre-Compensation Hiding: System Model and Experiment Design

This document describes the current codebase implementation of the RF fingerprint hiding experiment. It is based on the active Python modules, scripts, and YAML configuration files in this repository. The current implementation uses six transmitters, fixed hardware impairment parameters, and a residual-power upper bound of `0.1` for the default SNR=20 workflow.

## 1. Project Objective

The experiment studies transmitter-identity hiding in a multi-transmitter RF signal chain. The goal is to reduce the ability of a strong receiver-side classifier, named Eve, to infer the transmitter identity from received IQ samples, while keeping the communication quality within an acceptable range.

The system compares three received-signal classes:

| Method | Description |
|---|---|
| `uncompensated` | The clean 16-QAM signal is sent through transmitter hardware impairments and AWGN without any pre-compensation residual. |
| `random_perturb` | A random complex residual is added before transmission. Its per-sample power is matched to the fixed-precomp residual and then projected to the residual-power limit. |
| `fixed_precomp` | A fixed transmitter-side pre-compensation value is optimized once and applied before transmission. |

The main privacy metric is Eve's transmitter-identification accuracy. The communication metrics are BER and linear RMS EVM.

## 2. End-to-End Signal Model

For each sample, random bits are mapped to normalized Gray-coded 16-QAM symbols. The same clean symbol sequence is paired across all transmitters. Each transmitter has its own hardware impairment parameters, and the channel adds AWGN.

The high-level chain is:

```text
bits -> 16-QAM symbols x -> x + p_d -> transmitter impairment H_d(.; t) -> AWGN -> y_d
```

where:

| Symbol | Meaning |
|---|---|
| `D` | Number of transmitters. Current default: `6`. |
| `d` | Transmitter ID, `d in {0, 1, 2, 3, 4, 5}`. |
| `B` | Batch size. |
| `N` | Number of 16-QAM symbols per sample. Current default: `1024`. |
| `x` | Clean complex 16-QAM sequence, shape `[B, N]`. |
| `x_clean_tx` | Clean sequence repeated across transmitters, shape `[D, B, N]`. |
| `p_d` | Complex pre-compensation residual for transmitter `d`, shape `[D, B, N]` or flattened as `[D * B, N]`. |
| `H_d` | Transmitter-specific hardware impairment model. |
| `t` | Legacy time index retained in tensor formats; active hardware parameters are fixed. |
| `y_d` | Received complex signal after impairment and AWGN. |

The uncompensated baseline uses `p_d = 0`. The learned method uses `p_d` induced by the optimized fixed pre-compensation parameters.

## 3. Modulation Model

The code uses normalized Gray-coded 16-QAM. Every four bits form one symbol. The two I-axis bits and two Q-axis bits are mapped independently:

```text
00 -> -3
01 -> -1
11 ->  3
10 ->  1
```

The complex symbol is:

```text
x_n = (a_I + j a_Q) / sqrt(10)
```

This normalization gives unit average constellation power.

Implementation files:

| File | Role |
|---|---|
| `rfhide/modulation.py` | 16-QAM mapping, hard demapping, random bit generation. |
| `rfhide/metrics.py` | BER and EVM computation. |

## 4. AWGN Channel

The AWGN channel measures the input signal power and adds complex Gaussian noise according to the requested SNR:

```text
P_z = mean(|z_n|^2)
rho = 10^(SNR_dB / 10)
noise_power = P_z / rho
w_n ~ CN(0, noise_power)
y_n = z_n + w_n
```

The implementation supports scalar SNR and per-sample SNR vectors. In paired multi-transmitter batches, the same sample's SNR is shared by all transmitters.

Implementation file: `rfhide/channel.py`.

## 5. Hardware Impairment Model

Each transmitter is assigned fixed random hardware parameters. The current impairment bank supports gain, phase, carrier-frequency offset, IQ imbalance, and DC offset.

For transmitter `d` and time index `t`:

```text
G_d   = G_d0
phi_d = phi_d0
f_d   = f_d0
```

The main rotation and scaling are:

```text
r_d,n = 10^(G_d(t)/20) * u_d,n * exp(j * (phi_d(t) + 2*pi*f_d(t)*n/F_s))
```

IQ image leakage and DC offset are then added:

```text
y_d,n = r_d,n + image_coeff_d * conj(r_d,n) + dc_offset_d
```

### Current Impairment Parameters

These values are used by the current SNR=20 configuration and by the current multi-SNR workflow through `base_config: configs/snr20.yaml`.

| Parameter | Value |
|---|---:|
| Number of transmitters, `num_tx` | `6` |
| Sample rate | `1,000,000 Hz` |
| Gain standard deviation | `1.0 dB` |
| Gain drift | `0.0 dB/step` |
| Phase standard deviation | `0.2 rad` |
| Phase drift | `0.0 rad/step` |
| CFO standard deviation | `100.0 Hz` |
| CFO drift | `0.0 Hz/step` |
| IQ gain mismatch standard deviation | `0.5 dB` |
| IQ phase mismatch standard deviation | `0.05 rad` |
| Complex DC offset standard deviation | `0.02` per real/imag component |
| Maximum time index wrap | `100` |

Implementation file: `rfhide/impairments.py`.

## 6. Paired Multi-Transmitter Batch Design

The batch generator enforces a paired design so that transmitter identity cannot be inferred from different payload bits or different SNR values.

For each batch:

1. Generate `B` independent random bit sequences.
2. Map them to clean 16-QAM sequences `x_clean`.
3. Repeat the same clean sequences across all `D` transmitters.
4. Apply transmitter-specific impairment parameters.
5. Add AWGN.

Returned tensor shapes:

| Tensor | Shape | Meaning |
|---|---|---|
| `bits` | `[B, N * 4]` | Flattened 16-QAM bit labels. |
| `x_clean` | `[B, N]` complex | Clean 16-QAM symbols. |
| `x_clean_tx` | `[D, B, N]` complex | Same clean batch copied to all transmitters. |
| `tx_ids` | `[D]` | Transmitter IDs. |
| `snr_db` | `[B]` | Per-sample SNR, shared across transmitters. |
| `time_indices` | `[D, B]` | Legacy time-index tensor; no active drift is applied. |
| `y_imp` | `[D, B, N]` complex | Signal after hardware impairments, before AWGN. |
| `y_rx` | `[D, B, N]` complex | Signal after hardware impairments and AWGN. |

For fixed-SNR runs, all entries in `snr_db` equal the configured SNR. For a config containing `snr_list`, the generator samples an SNR per batch item, but the current multi-SNR experiment no longer trains a mixed-SNR model. Instead, it repeats the full single-SNR workflow independently for each SNR.

Implementation file: `rfhide/dataset.py`.

## 7. Feature Extraction for Distribution Alignment

The teacher uses raw received-signal features to align transmitter distributions. The feature extractor does not include transmitter IDs.

Given received signals `y` with shape `[D, B, N]`, the feature extractor uses:

| Component | Description |
|---|---|
| Downsampled real samples | `Re(y[..., ::downsample])` |
| Downsampled imaginary samples | `Im(y[..., ::downsample])` |
| Downsampled power | `|y[..., ::downsample]|^2` |
| Mean power | `mean(|y|^2)` |
| Power standard deviation | `std(|y|^2)` |
| Peak power | `max(|y|^2)` |

Default downsample stride is `4`. For `N=1024`, the default feature dimension is:

```text
256 real + 256 imag + 256 power + 3 summary statistics = 771
```

Implementation file: `rfhide/features.py`.

## 8. Teacher Compensation Model

The fixed pre-compensator is optimized offline and then used as the transmitter-side compensation value during evaluation.

Inputs:

| Input | Shape | Description |
|---|---|---|
| `x_clean` | `[B, N]` complex | Clean 16-QAM symbols. |
| `tx_ids` | `[D]` | Transmitter IDs. |
| `snr_db` | scalar or `[B]` | SNR condition. |
| `time_indices` | `[D, B]` or `[B]` | Drift time condition. |

Output:

```text
p_teacher: [D, B, N] complex
```

### Teacher Architecture

| Component | Current SNR=20 setting |
|---|---:|
| Number of transmitters | `6` |
| Tx embedding dimension | `16` |
| Scalar condition input | `[snr_db / 30, log1p(time_index) / 10]` |
| Scalar condition MLP | `Linear(2, 16) -> GELU -> Linear(16, 16)` |
| Signal input channels | `2 + condition_dim = 18` |
| Hidden channels | `48` |
| Residual Conv1D blocks | `3` |
| Residual block | `Conv1d(k=5) -> GELU -> Conv1d(k=5)` |
| Output layer | `Conv1d(k=3)` to two IQ channels |
| Output nonlinearity | `tanh` |
| Optimizer | AdamW |
| Learning rate | `0.001` |
| Epochs | `30` |
| Steps per epoch | `100` |
| Teacher batch size | `128` |
| Gradient clipping | max norm `5.0` |

Implementation file: `rfhide/models_teacher.py`.

### Residual Power Projection

The teacher output is projected so each residual obeys:

```text
mean(|p_d,b|^2) / mean(|x_b|^2) <= max_residual_power_ratio
```

Current `configs/snr20.yaml` value:

```text
max_residual_power_ratio = 0.1
```

The code uses the configured value when present. Some alternate fixed-SNR configs, such as `configs/snr15.yaml` and `configs/snr25.yaml`, currently still contain `0.05`. The current multi-SNR workflow uses `configs/snr20.yaml` as its base config, so its effective value is `0.1`.

### Teacher Training Loss

The current teacher loss is:

```text
L_teacher =
  lambda_align   * L_align
+ lambda_evm     * L_evm
+ lambda_softbit * L_softbit
+ lambda_power   * L_power
```

With current SNR=20 weights:

| Weight | Value |
|---|---:|
| `lambda_align` | `1.0` |
| `lambda_mean` | `1.0` |
| `lambda_cov` | `1.0` |
| `lambda_mmd` | `1.0` |
| `lambda_evm` | `5.0` |
| `lambda_softbit` | `2.0` |
| `lambda_power` | `10.0` |

The alignment term is:

```text
L_align = lambda_mean * L_mean + lambda_cov * L_cov + lambda_mmd * L_mmd
```

where all transmitter pairs are compared. With six transmitters, the pair set has `C(6, 2) = 15` pairs.

MMD uses a multi-kernel RBF with:

```text
mmd_sigmas = [0.5, 1.0, 2.0, 4.0, 8.0]
```

The communication terms are:

| Term | Implementation |
|---|---|
| `L_evm` | Linear RMS EVM between equalized received symbols and clean symbols. |
| `L_softbit` | Differentiable 16-QAM soft demapper with BCEWithLogits. |
| `L_power` | Hinge-like excess-power penalty above the configured residual power limit. |

The current code directly minimizes EVM and soft-bit loss; it has not yet been changed to a constraint-only communication penalty.

Main training script: `scripts/02_train_teacher_snr20.py`.

## 9. Offline Compensation Dataset

After fixed pre-compensation training, the repository can build an offline dataset for diagnostics. Each stored sample corresponds to one transmitter/sample pair, flattened from the original `[D, B, N]` output.

Stored columns:

| Key | Shape | Description |
|---|---|---|
| `x_clean` | `[S, N]` complex | Clean signal. |
| `bits` | `[S, N * 4]` | Flattened bit labels. |
| `tx_id` | `[S]` | Transmitter ID. |
| `snr_db` | `[S]` | SNR condition. |
| `time_index` | `[S]` | Drift time condition. |
| `p_teacher` | `[S, N]` complex | Teacher residual target. |
| `residual_power_ratio` | `[S]` | Residual-to-clean power ratio. |

For the current SNR=20 config:

| Split | Batches | Batch size | Samples with `D=6` |
|---|---:|---:|---:|
| Train | `8` | `64` | `8 * 64 * 6 = 3072` |
| Validation | `2` | `64` | `2 * 64 * 6 = 768` |
| Test | `2` | `64` | `2 * 64 * 6 = 768` |

Main script: `scripts/03_build_compensation_dataset.py`.

## 10. Fixed Pre-Compensation Model

The active workflow does not train a diffusion model. It optimizes fixed transmitter-side pre-compensation parameters once:

```text
p = f_theta(x_clean, tx_id)
```

It is implemented as a 1D conditional DDPM epsilon predictor over complex residuals represented as two real IQ channels.

### DDPM Forward Process

For teacher residual `p_0`, the forward process samples:

```text
p_t = sqrt(alpha_bar_t) * p_0 + sqrt(1 - alpha_bar_t) * epsilon
```

where complex Gaussian noise is generated as independent real and imaginary standard normal samples.

The training loss is epsilon prediction:

```text
L_DDPM = mean(|epsilon_pred - epsilon|^2)
```

### Fixed Precomp Architecture

| Component | Current SNR=20 setting |
|---|---:|
| Number of transmitters | `6` |
| Input signal channels | `4`: noisy residual I/Q plus clean signal I/Q |
| Condition dimension | `32` |
| Tx embedding dimension | `32` |
| Scalar condition input | `[snr_db / 30, log1p(time_index) / 10]` |
| Timestep embedding | Sinusoidal embedding, dimension `32` |
| Base channels | `64` |
| Residual Conv1D blocks | `4` |
| Residual block | `Conv1d(k=5) -> GroupNorm -> SiLU -> Conv1d(k=5)` |
| Output layer | `Conv1d(k=3)` to two IQ channels |
| Diffusion timesteps | Not used |
| Beta schedule | Linear from `0.0001` to `0.02` |
| Optimizer | AdamW |
| Learning rate | `0.0002` |
| Epochs | `50` |
| Batch size | `128` |
| Sample diagnostics interval | every `5` epochs |
| Gradient clipping | max norm `5.0` |
| Residual power projection | `max_residual_power_ratio = 0.1` in `configs/snr20.yaml` |

Sampling starts from complex Gaussian noise and runs the reverse DDPM update from `T-1` to `0`. The sampled residual is projected again to the configured residual-power bound.

Implementation file: `rfhide/fixed_precomp.py`.

Main training script: `scripts/02_train_teacher_snr20.py`.

## 11. Evaluation Signal Collection

The evaluation collector loads the optimized fixed-precomp checkpoint and creates three saved datasets:

| Dataset file | Method |
|---|---|
| `eval_uncompensated.pt` | No residual. |
| `eval_random_perturb.pt` | Random residual with power matched to fixed-precomp residual. |
| `eval_fixed_precomp.pt` | Fixed-precompensated residual. |

For each evaluation batch:

1. Generate a paired multi-transmitter batch.
2. Flatten `x_clean_tx`, `tx_ids`, `snr_db`, and `time_indices`.
3. Apply the fixed pre-compensation value.
4. Generate a random residual with matched per-sample residual power.
5. Apply the same hardware impairment bank to all three methods.
6. Equalize received samples for BER/EVM reporting.
7. Save real-imag channel tensors for Eve.

Saved evaluation columns:

| Key | Shape | Description |
|---|---|---|
| `signals` | `[S, 2, N]` | Received IQ channels. |
| `labels` | `[S]` | Transmitter IDs for Eve. |
| `bits` | `[S, N * 4]` | Bit labels. |
| `x_clean` | `[S, 2, N]` | Clean IQ channels. |
| `snr_db` | `[S]` | SNR values. |
| `tx_id` | `[S]` | Same as labels. |
| `evm` | `[S]` | Linear RMS EVM. |
| `ber` | `[S]` | Hard-decision BER. |
| `residual_power_ratio` | `[S]` | Residual-to-clean power ratio. |

Current SNR=20 eval collection:

| Parameter | Value |
|---|---:|
| Eval batch size | `32` |
| Number of eval batches | `6` |
| Samples per method with `D=6` | `6 * 32 * 6 = 1152` |

Main script: `scripts/05_collect_eval_signals_snr20.py`.

## 12. Strong Eve-2 Classifier

Eve is trained as a fresh classifier separately for each signal class. This creates a strong attacker setting because Eve adapts to each evaluated distribution rather than reusing one fixed classifier.

### Eve Architecture

| Layer | Setting |
|---|---|
| Input | `[B, 2, N]` real-imag channels |
| Conv1 | `2 -> 32`, kernel size `7`, padding `3` |
| Norm/activation | BatchNorm1d + ReLU |
| Conv2 | `32 -> 64`, kernel size `5`, padding `2` |
| Norm/activation | BatchNorm1d + ReLU |
| Conv3 | `64 -> 128`, kernel size `5`, padding `2` |
| Norm/activation | BatchNorm1d + ReLU |
| Pool | Adaptive average pool to length `1` |
| Embedding | 128-dimensional pooled feature |
| Classifier | `Linear(128, 6)` |

### Eve Training Parameters

| Parameter | Value |
|---|---:|
| Epochs | `30` |
| Batch size | `64` |
| Optimizer | AdamW |
| Learning rate | `0.001` |
| Train/validation/test split | `60% / 20% / 20%` |
| Split policy | Balanced by transmitter label |
| Classes | `6` |

Main script: `scripts/06_train_eve_eval_snr20.py`.

## 13. Plotting and Reporting

The SNR=20 plotting script reads saved evaluation datasets and Eve results, then produces:

| Output | Meaning |
|---|---|
| `accuracy_comparison_snr20.png` | Eve test accuracy for the three methods. |
| `ber_comparison_snr20.png` | Mean BER for the three methods. |
| `evm_comparison_snr20.png` | Mean EVM for the three methods. |
| `tsne_uncompensated_snr20.png` | t-SNE for uncompensated signals. |
| `tsne_random_perturb_snr20.png` | t-SNE for random perturbation. |
| `tsne_fixed_precomp_snr20.png` | t-SNE for fixed pre-compensation. |
| `tsne_all_methods_snr20.png` | Combined t-SNE across methods. |
| `final_summary_snr20.csv/json` | Numeric summary. |

The t-SNE feature source is Eve's 128-dimensional embedding if the corresponding Eve checkpoint exists; otherwise, the script falls back to flattened IQ samples. The maximum combined t-SNE sample count defaults to `900`.

Main script: `scripts/07_plot_snr20_results.py`.

## 14. Multi-SNR Experiment Design

The current multi-SNR implementation intentionally repeats the complete SNR=20 workflow independently for each SNR. It does not train a single mixed-SNR fixed pre-compensator.

The controlling config is `configs/multisinr.yaml`:

```yaml
experiment:
  output_dir: outputs/multisnr

base_config: configs/snr20.yaml

signal:
  snr_list: [0, 5, 10, 15, 20, 25, 30]
```

For each SNR value, `scripts/08_train_multisnr.py` derives a single-SNR config from `configs/snr20.yaml`, changes only:

| Field | Per-SNR value |
|---|---|
| `signal.snr_db` | Current SNR from `snr_list` |
| `experiment.output_dir` | `outputs/multisnr/snr{tag}` |
| Checkpoint/data paths | Paths under the per-SNR output directory |

Then it runs:

```text
02_train_teacher_snr20.py
03_build_compensation_dataset.py
```

For evaluation, `scripts/09_eval_multisnr.py` runs:

```text
05_collect_eval_signals_snr20.py
06_train_eve_eval_snr20.py
```

for each SNR, then aggregates the results into:

| Output | Location |
|---|---|
| Per-SNR outputs | `outputs/multisnr/snr{tag}/...` |
| Per-SNR three-panel t-SNE | `outputs/multisnr/snr{tag}/figures/tsne_methods_snr{tag}.png` |
| Aggregate accuracy curve | `outputs/multisnr/figures/accuracy_vs_snr.png` |
| Aggregate BER curve | `outputs/multisnr/figures/ber_vs_snr.png` |
| Aggregate EVM curve | `outputs/multisnr/figures/evm_vs_snr.png` |
| Aggregate residual-power curve | `outputs/multisnr/figures/residual_power_vs_snr.png` |
| Aggregate numeric table | `outputs/multisnr/logs/multisnr_results.csv/json` |

The per-SNR t-SNE figure is a single horizontal figure with three subplots: uncompensated, random perturbation, and fixed pre-compensation.

## 15. Current Main Configuration Summary

The current default single-SNR configuration is `configs/snr20.yaml`.

### General

| Parameter | Value |
|---|---:|
| Experiment name | `snr20_minimal_loop` |
| Output directory | `outputs/snr20` |
| Seed | `42` |
| Preferred device | CUDA if available |
| SNR | `20 dB` |
| Sample rate | `1 MHz` |
| Number of symbols | `1024` |
| Modulation | `16qam` |
| Data batch size | `32` |
| Number of workers | `0` |
| Max time index | `100` |

### Hardware Impairments

| Parameter | Value |
|---|---:|
| `num_tx` | `6` |
| `gain_db_std` | `1.0` |
| `gain_drift_db_per_step` | `0.0` |
| `phase_rad_std` | `0.2` |
| `phase_drift_rad_per_step` | `0.0` |
| `cfo_hz_std` | `100.0` |
| `cfo_drift_hz_per_step` | `0.0` |
| `iq_gain_mismatch_db_std` | `0.5` |
| `iq_phase_mismatch_rad_std` | `0.05` |
| `dc_offset_std` | `0.02` |

### Teacher

| Parameter | Value |
|---|---:|
| `epochs` | `30` |
| `batch_size` | `128` |
| `lr` | `0.001` |
| `num_steps_per_epoch` | `100` |
| `hidden_channels` | `48` |
| `num_blocks` | `3` |
| `condition_dim` | `16` |
| `max_residual_power_ratio` | `0.1` |
| `lambda_align` | `1.0` |
| `lambda_mean` | `1.0` |
| `lambda_cov` | `1.0` |
| `lambda_mmd` | `1.0` |
| `lambda_evm` | `5.0` |
| `lambda_softbit` | `2.0` |
| `lambda_power` | `10.0` |
| `mmd_sigmas` | `[0.5, 1.0, 2.0, 4.0, 8.0]` |

### Compensation Dataset

| Parameter | Value |
|---|---:|
| `batch_size` | `64` |
| `train_batches` | `8` |
| `val_batches` | `2` |
| `test_batches` | `2` |
| `checkpoint` | `outputs/snr20/checkpoints/teacher_best.pt` |

### Fixed Precomp

| Parameter | Value |
|---|---:|
| `checkpoint` | `outputs/snr20/checkpoints/teacher_best.pt` |
| `epochs` | `50` |
| `batch_size` | `128` |
| `lr` | `0.0002` |
| `base_channels` | `64` |
| `condition_dim` | `32` |
| `num_blocks` | `4` |
| `beta_start` | `0.0001` |
| `beta_end` | `0.02` |
| `max_residual_power_ratio` | `0.1` |
| `sample_every` | `5` |
| `train_data` | `outputs/snr20/data/comp_train.pt` |
| `val_data` | `outputs/snr20/data/comp_val.pt` |

### Evaluation and Eve

| Parameter | Value |
|---|---:|
| Eval collection batch size | `32` |
| Eval collection batches | `6` |
| Fixed precomp checkpoint | `outputs/snr20/checkpoints/teacher_best.pt` |
| Eve epochs | `30` |
| Eve batch size | `64` |
| Eve learning rate | `0.001` |
| Train ratio | `0.6` |
| Validation ratio | `0.2` |
| Test ratio | `0.2` |
| Eve classes | `6` |
| Eve max batches | `null` |

## 16. Execution Commands

### Single SNR=20 Pipeline

```powershell
python scripts\00_smoke_test.py --config configs\snr20.yaml
python scripts\01_check_signal_chain.py --config configs\snr20.yaml
python scripts\02_train_teacher_snr20.py --config configs\snr20.yaml
python scripts\03_build_compensation_dataset.py --config configs\snr20.yaml
python scripts\05_collect_eval_signals_snr20.py --config configs\snr20.yaml
python scripts\06_train_eve_eval_snr20.py --config configs\snr20.yaml
python scripts\07_plot_snr20_results.py --config configs\snr20.yaml
```

### Multi-SNR Pipeline

```powershell
python scripts\08_train_multisnr.py --config configs\multisinr.yaml
python scripts\09_eval_multisnr.py --config configs\multisinr.yaml
```

The multi-SNR pipeline uses `configs/snr20.yaml` as its base template unless overridden with `--base-config`.

## 17. Verification Coverage

The `tests/` directory validates the main components:

| Test file | Coverage |
|---|---|
| `test_imports.py` | Import sanity checks. |
| `test_signal_chain.py` | 16-QAM, AWGN, and hardware impairment behavior. |
| `test_dataset.py` | Paired multi-transmitter batch generation. |
| `test_losses.py` | Feature extraction and alignment/communication losses. |
| `test_teacher_model.py` | Teacher output shape and conditioning behavior. |
| `test_compensation_dataset.py` | Offline compensation dataset structure and balance. |
| `test_fixed_precomp_model.py` | Fixed precomp conditioning, time-invariance, and residual projection. |
| `test_eve_model.py` | Eve CNN shape, training, and label behavior. |
| `test_eval_signal_collection.py` | Saved evaluation signal dataset consistency. |

Note that several tests still use compact synthetic settings or locally defined `num_tx=3` fixtures for unit testing. The active experiment configurations now use `num_tx=6`.

## 18. Important Implementation Notes

1. The current SNR=20 workflow is the source of truth for the complete experimental procedure.
2. The multi-SNR workflow repeats that complete SNR=20-style procedure independently for each SNR.
3. The current SNR=20 residual-power upper bound is `0.1`.
4. The current teacher still directly minimizes EVM and soft-bit losses. A constraint-only communication penalty would require a future code change.
5. The paper draft under `paper/` contains older, partially garbled, and currently outdated values such as three transmitters and a `0.05` residual-power bound. This document follows the active code and configuration files instead.
