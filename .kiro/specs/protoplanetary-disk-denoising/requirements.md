w# Requirements Document

## Introduction

This document defines the requirements for a machine learning denoising pipeline targeting astronomical observations of protoplanetary disks. The pipeline uses Denoising Diffusion Probabilistic Models (DDPM) trained on synthetic observations generated from hydrodynamic simulations and radiative transfer codes. The goal is to replace or augment traditional denoising methods (e.g., the CLEAN algorithm) with a faster, more accurate ML-based approach that generalizes to real ALMA and VLT observations, as well as to line emission data and other telescope formats.

The system is implemented in Python and PyTorch, with C/Fortran components for simulation-side data generation. It must support the full lifecycle: data ingestion, preprocessing, augmentation, model training, inference, evaluation, and scientific validation.

---

## Glossary

- **Pipeline**: The end-to-end system encompassing data preparation, model training, inference, and evaluation.
- **DDPM**: Denoising Diffusion Probabilistic Model — a generative model that learns to reverse a noise-addition process to recover clean images.
- **DDIM**: Denoising Diffusion Implicit Model — an accelerated sampling variant of DDPM that reduces inference steps without retraining.
- **U-Net**: A convolutional encoder-decoder architecture with skip connections used as the backbone of the DDPM.
- **Dirty_Image**: A noisy synthetic or real observation used as model input (e.g., `dirty.npy` or a FITS file from ALMA/VLT).
- **Clean_Image**: The ground-truth noiseless image used as training target (e.g., `clean.npy` from radiative transfer simulation output).
- **FITS**: Flexible Image Transport System — the standard file format for astronomical data.
- **CLEAN**: The traditional radio interferometric denoising algorithm used as a baseline for comparison.
- **Synthetic_Observation**: A simulated observation produced by applying a telescope response model to hydrodynamic simulation output.
- **Data_Pipeline**: The subsystem responsible for loading, preprocessing, and augmenting training data.
- **Trainer**: The subsystem responsible for executing the training loop, logging, and checkpointing.
- **Inference_Engine**: The subsystem responsible for applying a trained model to new observations.
- **Evaluator**: The subsystem responsible for computing quantitative metrics and scientific validation.
- **Augmentor**: The subsystem responsible for applying physics-aware data augmentation transforms.
- **PSNR**: Peak Signal-to-Noise Ratio — a standard image quality metric.
- **SSIM**: Structural Similarity Index — a perceptual image quality metric.
- **Noise_Schedule**: The sequence of noise levels (β_t) used in the forward diffusion process.
- **Timestep**: An integer index t ∈ [0, T] representing a point in the diffusion process.
- **Domain_Adaptation**: Techniques for reducing the distribution gap between synthetic training data and real observational data.

---

## Requirements

### Requirement 1: Data Ingestion

**User Story:** As a researcher, I want to load paired noisy and clean synthetic observations from multiple file formats, so that I can build a training dataset from simulation outputs.

#### Acceptance Criteria

1. THE Data_Pipeline SHALL load paired Dirty_Image and Clean_Image arrays from `.npy` files into PyTorch tensors with shape `(N, C, H, W)`.
2. THE Data_Pipeline SHALL load single-channel astronomical images from FITS files, preserving the WCS header metadata for downstream use.
3. WHEN a FITS file contains multiple HDU extensions, THE Data_Pipeline SHALL allow the user to specify which HDU index to read.
4. IF a file path does not exist or the file is corrupt, THEN THE Data_Pipeline SHALL raise a descriptive error identifying the file and the failure mode.
5. THE Data_Pipeline SHALL support images with spatial dimensions that are not powers of two by padding them to the nearest valid model input size.
6. WHEN loading data, THE Data_Pipeline SHALL report the number of samples loaded, the image dimensions, and the intensity range.

---

### Requirement 2: Data Preprocessing

**User Story:** As a researcher, I want images to be normalized and resized consistently before training, so that the model receives well-conditioned inputs regardless of the original data scale.

#### Acceptance Criteria

1. THE Data_Pipeline SHALL normalize image intensities to the range [−1, 1] using per-image min-max normalization before passing them to the model.
2. THE Data_Pipeline SHALL pad images to a spatial size that is a multiple of 16 using zero-padding, and record the padding dimensions to enable exact reversal during postprocessing.
3. WHEN postprocessing model outputs, THE Inference_Engine SHALL remove the padding applied during preprocessing and restore the original spatial dimensions.
4. WHEN postprocessing model outputs, THE Inference_Engine SHALL invert the normalization transform to restore physical intensity units.
5. THE Data_Pipeline SHALL apply identical preprocessing transforms to both the Dirty_Image and Clean_Image within each training pair to preserve their correspondence.

---

### Requirement 3: Physics-Aware Data Augmentation

**User Story:** As a researcher, I want training data augmented with physically valid transforms, so that the model learns rotation and reflection invariances present in protoplanetary disk observations.

#### Acceptance Criteria

1. THE Augmentor SHALL apply random rotations in multiples of 90 degrees to training image pairs, applying the same rotation to both the Dirty_Image and Clean_Image in each pair.
2. THE Augmentor SHALL apply random horizontal and vertical flips to training image pairs, applying the same flip to both images in each pair.
3. WHERE Gaussian noise augmentation is enabled, THE Augmentor SHALL add zero-mean Gaussian noise with a configurable standard deviation to Dirty_Images only, leaving Clean_Images unmodified.
4. THE Augmentor SHALL NOT apply augmentations that alter the physical flux scale of the images (e.g., brightness scaling that is not flux-conserving).
5. THE Augmentor SHALL apply augmentations only during training and SHALL NOT apply them during validation or inference.
6. THE Data_Pipeline SHALL expose a configuration parameter to enable or disable each augmentation type independently.

---

### Requirement 4: DDPM Model Architecture

**User Story:** As a researcher, I want a DDPM model with a U-Net backbone, so that the model can learn to denoise astronomical images by reversing a learned diffusion process.

#### Acceptance Criteria

1. THE DDPM SHALL implement a forward diffusion process that adds Gaussian noise to a Clean_Image at a given Timestep t according to the closed-form expression: `x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 − ᾱ_t) * ε`, where `ε ~ N(0, I)`.
2. THE DDPM SHALL implement a configurable Noise_Schedule supporting both linear and cosine beta schedules over a configurable number of Timesteps T (default T = 1000).
3. THE U-Net SHALL accept a noisy image tensor of shape `(B, C, H, W)` and a Timestep embedding as inputs, and SHALL output a predicted noise tensor of the same shape `(B, C, H, W)`.
4. THE U-Net SHALL include sinusoidal positional time embeddings that encode the Timestep t and inject it into each residual block of the encoder and decoder.
5. THE U-Net SHALL include self-attention blocks at configurable resolution levels to capture long-range spatial dependencies.
6. THE U-Net SHALL use skip connections between corresponding encoder and decoder levels to preserve spatial detail.
7. THE DDPM SHALL implement a reverse diffusion sampling procedure that iteratively denoises a noisy input from Timestep T to Timestep 0 to produce a clean output image.
8. WHEN the number of model parameters is logged, THE U-Net SHALL report the total parameter count to enable reproducibility documentation.

---

### Requirement 5: Conditional Denoising

**User Story:** As a researcher, I want the model to condition on the noisy observation during inference, so that the denoised output is guided by the actual observed data rather than generated from pure noise.

#### Acceptance Criteria

1. THE DDPM SHALL support a conditional inference mode in which the Dirty_Image is used to initialize the reverse diffusion process rather than pure Gaussian noise.
2. WHERE concatenation conditioning is enabled, THE U-Net SHALL accept the Dirty_Image concatenated channel-wise with the noisy latent as input, increasing the input channel count accordingly.
3. THE Pipeline SHALL expose a configuration parameter to select the conditioning strategy (concatenation or initialization-only).
4. WHEN conditional inference is used, THE Inference_Engine SHALL produce outputs that are consistent with the spatial structure of the input Dirty_Image.

---

### Requirement 6: Training Pipeline

**User Story:** As a researcher, I want a reproducible training pipeline with logging and checkpointing, so that I can monitor training progress and resume interrupted runs.

#### Acceptance Criteria

1. THE Trainer SHALL train the DDPM by sampling a random Timestep t uniformly from [0, T] for each training example, computing the forward diffusion, predicting the noise with the U-Net, and minimizing the mean squared error between the predicted and true noise.
2. THE Trainer SHALL use the AdamW optimizer with configurable learning rate, weight decay, and beta parameters.
3. THE Trainer SHALL apply a cosine annealing learning rate schedule with a configurable linear warmup period.
4. THE Trainer SHALL log training loss, validation loss, PSNR, and SSIM to Weights & Biases at a configurable logging interval.
5. THE Trainer SHALL save a model checkpoint whenever the validation PSNR improves, and SHALL also save the most recent checkpoint to enable training resumption.
6. WHEN a checkpoint path is provided at startup, THE Trainer SHALL resume training from that checkpoint, restoring model weights, optimizer state, and the epoch counter.
7. THE Trainer SHALL generate and log sample denoised images to Weights & Biases at a configurable sampling interval during training.
8. THE Trainer SHALL support mixed-precision (float16) training on CUDA devices to reduce memory usage and increase throughput.
9. THE Trainer SHALL accept a configuration file (YAML or dataclass) that specifies all hyperparameters, and SHALL log the full configuration at the start of each run.

---

### Requirement 7: Fast Inference Sampling

**User Story:** As a researcher, I want a fast sampling mode for inference, so that I can denoise new observations in a practical amount of time without retraining the model.

#### Acceptance Criteria

1. THE Inference_Engine SHALL implement DDIM sampling that produces a denoised image using a configurable number of steps S (default S = 50), where S < T.
2. WHEN DDIM sampling is used with S = 50 steps, THE Inference_Engine SHALL produce a denoised image in no more than 10× the wall-clock time required for a single U-Net forward pass on the same hardware.
3. THE Inference_Engine SHALL support both DDPM (full T-step) and DDIM (S-step) sampling modes, selectable via configuration.
4. THE Inference_Engine SHALL accept a FITS file as input, apply preprocessing, run reverse diffusion, apply postprocessing, and write the denoised result as a FITS file preserving the original WCS header.
5. THE Inference_Engine SHALL support batch processing of all FITS files in a specified input directory, writing results to a specified output directory.

---

### Requirement 8: Evaluation Metrics

**User Story:** As a researcher, I want quantitative metrics computed on held-out data, so that I can objectively assess and compare denoising performance.

#### Acceptance Criteria

1. THE Evaluator SHALL compute PSNR between the denoised output and the Clean_Image ground truth for each validation sample.
2. THE Evaluator SHALL compute SSIM between the denoised output and the Clean_Image ground truth for each validation sample.
3. THE Evaluator SHALL compute the mean and standard deviation of PSNR and SSIM across the validation set and report them as summary statistics.
4. THE Evaluator SHALL measure and report the wall-clock inference time per image for each evaluated method.
5. THE Evaluator SHALL compute a flux conservation score defined as the ratio of total integrated flux in the denoised image to total integrated flux in the Clean_Image, and SHALL report the deviation from 1.0.
6. THE Evaluator SHALL compute a structural feature preservation score by comparing the locations and relative intensities of the top-k intensity peaks between the denoised output and the Clean_Image.

---

### Requirement 9: Baseline Comparison

**User Story:** As a researcher, I want the pipeline to compare DDPM denoising against traditional methods, so that I can quantify the improvement over the current state of the art.

#### Acceptance Criteria

1. THE Evaluator SHALL apply a Gaussian filter baseline to each test image and compute PSNR and SSIM for the result.
2. THE Evaluator SHALL apply a median filter baseline to each test image and compute PSNR and SSIM for the result.
3. WHERE a CLEAN algorithm implementation is available, THE Evaluator SHALL apply it to each test image and compute PSNR and SSIM for the result.
4. THE Evaluator SHALL produce a summary table comparing PSNR, SSIM, flux conservation score, and inference time across all evaluated methods.
5. THE Evaluator SHALL generate side-by-side visualization figures showing the Dirty_Image, Clean_Image ground truth, DDPM output, and at least one traditional baseline for a representative set of test images.

---

### Requirement 10: Generalization to Line Emission Data

**User Story:** As a researcher, I want the pipeline to support line emission data cubes in addition to continuum images, so that the model can be applied to a broader range of observational data products.

#### Acceptance Criteria

1. THE Data_Pipeline SHALL load 3D spectral line data cubes from FITS files with shape `(N_channels, H, W)` and SHALL support processing individual channel maps as 2D images.
2. THE Data_Pipeline SHALL expose a configuration parameter to select between continuum (2D) and spectral line (3D) data modes.
3. WHEN operating in spectral line mode, THE Inference_Engine SHALL process each spectral channel independently and reassemble the denoised cube in the original channel order.
4. THE Evaluator SHALL compute PSNR and SSIM per spectral channel and SHALL report the mean and standard deviation across channels for line emission data.

---

### Requirement 11: Domain Adaptation for Real Observations

**User Story:** As a researcher, I want the model to generalize from synthetic training data to real ALMA and VLT observations, so that the pipeline produces scientifically useful results on actual telescope data.

#### Acceptance Criteria

1. THE Data_Pipeline SHALL support loading real observational FITS files from ALMA and VLT alongside synthetic training data.
2. THE Augmentor SHALL support adding synthetic telescope noise patterns to Clean_Images during training to simulate the noise characteristics of real observations.
3. THE Trainer SHALL support a domain adaptation training mode in which the model is fine-tuned on a small set of real observations after initial training on synthetic data.
4. WHEN evaluated on real ALMA observations, THE Evaluator SHALL produce qualitative comparison figures showing the Dirty_Image and the DDPM-denoised output for visual scientific assessment.
5. THE Pipeline SHALL preserve all FITS header metadata (including WCS, beam parameters, and observation metadata) in denoised output files to maintain scientific usability.

---

### Requirement 12: Dataset Curation and Public Release

**User Story:** As a researcher, I want the training dataset and preprocessing pipeline to be documented and publicly releasable, so that other researchers can reproduce results and build on this work.

#### Acceptance Criteria

1. THE Data_Pipeline SHALL include a script that downloads or generates the synthetic training dataset and organizes it into a standardized directory structure.
2. THE Data_Pipeline SHALL produce a dataset manifest file listing each training sample with its file path, image dimensions, intensity range, and split assignment (train/validation/test).
3. THE Pipeline SHALL include a data card document describing the dataset provenance, simulation parameters, telescope response model, and known limitations.
4. THE Data_Pipeline SHALL implement a deterministic train/validation/test split using a configurable random seed to ensure reproducibility.
5. THE Data_Pipeline SHALL verify dataset integrity by checking that each Dirty_Image has a corresponding Clean_Image of identical spatial dimensions, and SHALL report any mismatches.
6. FOR ALL valid Dirty_Image and Clean_Image pairs in the dataset, THE Data_Pipeline SHALL confirm that the pair was generated from the same simulation snapshot (round-trip provenance check).

---

### Requirement 13: Model Documentation and Reproducibility

**User Story:** As a researcher, I want the model architecture, training procedure, and results to be fully documented, so that the work can be reproduced and submitted for scientific publication.

#### Acceptance Criteria

1. THE Pipeline SHALL include a README documenting installation, data preparation, training, inference, and evaluation steps with example commands.
2. THE Trainer SHALL log the full model architecture summary (layer names, shapes, parameter counts) at the start of each training run.
3. THE Pipeline SHALL include a configuration file for each reported experiment that fully specifies the hyperparameters used to produce the result.
4. THE Pipeline SHALL include a results notebook that loads a trained checkpoint, runs evaluation on the test set, and renders all comparison figures and metric tables.
5. THE Pipeline SHALL pin all Python dependency versions in a `requirements.txt` or `pyproject.toml` to ensure environment reproducibility.
