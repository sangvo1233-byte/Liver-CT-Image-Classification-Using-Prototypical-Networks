# Liver CT Image Classification Using Prototypical Networks

Research prototype for two-stage image classification: **no tumor / tumor**, followed by **benign / malignant**. The model classifies 2D images; it does not output lesion boxes or masks.

This implementation addresses grouped data splits, consistent inference, matched-label baselines, and seed/error analysis. The **exploratory benchmark is complete** for all three methods, K=5/10/30 and seeds 42/43/44: 27 paired configurations and 81 sets of stage/pipeline metrics. See [the experiment report](EXPERIMENTS.md) and [published aggregate results](results/benchmark-20261001/summary.csv).

Patient identifiers and original/augmentation relationships remain unknown, and annotation records are unavailable. Results are technical comparisons within the available image collection, not verified clinical performance. Older files in `results/metrics` and `results/figures` are historical artifacts of the previous protocol.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\Activate.ps1
```

Use the PyTorch build appropriate to your machine if GPU acceleration is needed. The code also runs on CPU. Data and legacy weights are not included in Git; the old download placeholder is not a working source.

The reported run used the official CUDA 12.1 wheels. To reproduce that environment,
install them in the venv before installing `requirements.txt`:

```powershell
.\.venv\Scripts\python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

Other CPU/GPU builds are listed in the [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/#v251).

## 1. Inventory and prepare data

Keep source images unchanged. Expected input folders are `train/benign`, `train/malignant`, `train_presence/no_tumor`, `train_presence/tumor`, and optional `test/<class>`. `malignant_extra` is inventoried as malignant, subject to label verification.

```powershell
python prepare_data.py --data-root data --out prepared/inventory
```

The command writes `inventory.csv`, `metadata_template.csv`, and `audit.json`. It checks readability, decoded-image duplicates, conflicting labels and existing test overlap. Hashes detect identical pixels; they cannot establish patient identity or discover every augmented variant.

The metadata CSV needs:

| Field | Meaning |
|---|---|
| path | Relative image path, matching the inventory |
| label | no_tumor, tumor, benign or malignant; verified from the source |
| patient_id | Patient/case identifier from source records, not an invented ID per file |
| source_image_id | Original image identifier shared with all its derived variants; globally unique |
| is_original | true for an original image, false for an offline augmented variant |
| label_verified | true when source/annotation records have been checked; false when that verification is unavailable |

Patient IDs must be globally unique across data sources, e.g. namespace them by dataset. Obtain these fields from original records/export mappings. Renamed files alone do not establish provenance. Do not mark unknown images as verified/original just to pass the gate.

```powershell
python prepare_data.py --data-root data --metadata metadata.csv --out prepared/split
```

The default preserves the existing `test/` partition and splits the remaining groups approximately 80/20 into train/validation. Existing test overlap with development causes an error. All images sharing a patient, original image or decoded pixels remain together across both tasks. Validation/test exclude offline augmented variants. All three typed labels must be represented in each partition.

If there is no usable existing test, `--resplit-all` explicitly requests an exploratory grouped 70/15/15 split of the available pool. It does not turn previously used data into a fresh external test. If patient IDs are irrecoverable, `--group-by source_image_id` allows an explicit weaker split; the corresponding benchmark also requires `--allow-image-groups` and is labeled exploratory.

When annotation records cannot be verified, keep `label_verified=false` and explicitly opt into `--allow-unverified-labels` in **both** preparation and experiments. This permits a technical comparison against the provided labels. It records `label_provenance=UNVERIFIED` in audits, run configurations, models and metrics, and marks real runs exploratory. The flag still requires known original/augmentation relationships and does not remove any split or duplicate checks. A hospital name or a missing certificate alone cannot establish whether the labels are correct or incorrect.

```powershell
python prepare_data.py --data-root data --metadata metadata.csv --out prepared/exploratory --group-by source_image_id --allow-unverified-labels
python run_experiments.py --manifest prepared/exploratory/manifest.csv --data-root data --out runs/exploratory --allow-image-groups --allow-unverified-labels
```

For this project, the author reports that data came from Hospital K and labels were assigned by a physician. Patient identifiers and supporting annotation/source records are currently unavailable. These are author-reported provenance statements, not independent institutional confirmation. Describe outcomes as agreement with supplied image labels; patient independence and clinical validity remain unverified.

If both patient IDs and original/augmentation relationships are irrecoverable, explicitly use **current-file exploratory evaluation**:

```powershell
python prepare_data.py --data-root data --out prepared/current_files --exploratory-files
python run_experiments.py --manifest prepared/current_files/manifest.csv --data-root data --out runs/current_files --exploratory-files
```

The recorded complete grid used `--steps 500 --val-every 10 --patience 10 --batch-size 32 --bootstrap 200 --threads 4 --device cuda`. The exact command and settings are in [EXPERIMENTS.md](EXPERIMENTS.md). If using a run directory other than `runs/benchmark`, update both paths in `app/config.yaml` accordingly.

This mode groups identical decoded pixels across all task folders and retains known split boundaries. Missing patient/source IDs remain blank, `is_original=unknown` and `label_verified=false`; no patient or original-image identity is invented. Checkpoint/threshold selection still uses validation only. Exact duplicates, incompatible labels, path escapes and modified source pixels are still rejected.

The only available independence check is exact decoded pixels. Different augmentations and different slices of one patient can still cross partitions. Validation/test may contain old augmented images because their ancestry is unknown. Thus **K means selected unique-pixel training images per class**, not independent original images or patients. Results, artifacts, summaries and plot axes use that unit, and real runs carry `EXPLORATORY_UNKNOWN_ANCESTRY_AND_UNVERIFIED_LABELS`. Bootstrap intervals are image-group intervals, not patient intervals; repeated seeds cannot repair unknown leakage. Such results describe a technical comparison within the available image collection and cannot establish patient generalization, clinical accuracy or verified label efficiency.

Every output directory must be new. Source files are never overwritten or removed.

### Recovered augmentation settings

`augmentation.yaml` records the actual offline recipe in notebooks 01/02: rotation ±25 degrees, horizontal flip probability 0.5, brightness factor 0.7–1.3, OpenCV linear interpolation and reflected borders. Shift/zoom/shear were declared but not applied. The old balancing targets were 317/class for subtype and 300/class for presence. The public v3 training augmentation and its seed 42 are recorded separately.

No offline augmentation seed was found in notebooks 01/02. **Seed 42 in the new generator is a new reproducibility choice.** It cannot restore deleted images or the exact historical augmented dataset. Reproducing historical training images would also require the original inputs, processing/order information, RNG state and compatible libraries.

```powershell
python augment_train.py --manifest prepared/split/manifest.csv --data-root data --out runs/augmented_train --config augmentation.yaml --seed 42
```

In manifest mode the generator takes only known original **training** images, writes PNG variants into a new directory outside source data, and records source paths/IDs, labels, pixel/file hashes, per-image seeds and sampled parameters in `metadata.csv`, with versions and configuration in `audit.json`. It inherits label provenance and patient/source groups; it does not invent missing identifiers or apply Otsu cropping again. Output contains augmented variants only, not a replacement original dataset. Missing original/augmentation ancestry remains unresolved.

For a technical preview from a remaining file whose ancestry is unknown:

```powershell
python augment_train.py --input-path train/benign/benign_001.jpg --data-root data --out runs/current_file_preview --seed 42
```

This mode records `source_mode=CURRENT_FILE_UNKNOWN_ANCESTRY` and `source_is_original=unknown`; patient/original-image IDs remain blank and labels remain unverified. A hash identifies the current file's pixels, not a patient or a recovered original. These preview outputs cannot pass the benchmark metadata gate as independent originals.

The main benchmark already augments its selected training examples online. Offline augmentation is optional and does not create additional independent labeled samples. Keep all derivatives of an original in the same group and keep val/test unchanged.

## 2–4. Baselines, inference and error analysis

```powershell
python run_experiments.py --manifest prepared/split/manifest.csv --data-root data --out runs/benchmark
```

Defaults: label budgets 5/10/30 independent original samples per class, seeds 42/43/44, and the following methods on both tasks:

- `frozen`: ImageNet ResNet-34 features and class prototypes, without encoder training.
- `finetune`: ImageNet ResNet-34 fine-tuned with a binary classifier and cross-entropy loss.
- `protonet`: ImageNet ResNet-34 with the v3 projection head, trained episodically with squared Euclidean prototype logits.

In the metadata-based protocol, within each task/budget/seed all methods receive exactly the same labeled original samples. There is one sample per independent group. Groups that contain both positive and negative slices are allocated without using one patient in both classes of the selected budget. Both trained methods use the same sampled image indices, augmentation policy, optimizer settings and maximum step count; random augmentation draws need not match exactly between architectures. Separate models are fitted for presence and subtype; their combined label cost is recorded in their individual artifacts and is not claimed to be only K labels for the whole pipeline. Current-file exploratory mode instead selects unique-pixel images, with independence unverified.

The encoder may only see the selected training labels. For episodic training, support and query are disjoint, with actual counts capped by the available labeled samples. For inference, prototype methods use all selected training support samples; this is K per class, never the entire unlabeled training pool.

These are **training-label budgets**. Labels for the fixed validation/test sets are additional; the validation label count is saved in each stage artifact. Report these costs separately when discussing label efficiency.

Preprocessing is shared: resize, RGB conversion and ImageNet normalization. No automatic Otsu crop is applied. Use consistently prepared inputs, or start from verified original CT images and assess this full-image protocol separately. Any preprocessing change requires retraining and reselecting validation thresholds.

Checkpoint selection uses validation macro-F1 at threshold 0.5. After selecting the checkpoint, its threshold is chosen **only on validation**, maximizing precision subject to a validation recall target of 0.90. The target is not a guarantee of test recall. There is no threshold selection on test. The same locked model, support prototypes, preprocessing and class mapping are used by the app.

Each run saves:

- `config.json`: dataset manifest hash, source-code hashes, settings, evidence status and runtime versions.
- `<method>_k<K>_seed<S>/<task>/model.pt`: weights, locked prototypes, support paths/hashes, class mapping and validation threshold.
- Per-stage `history.csv`, `metrics.json`, `predictions.csv`, `errors.csv` and an error gallery when errors exist.
- Pipeline metrics, three-class confusion matrix and error-stage attribution. A malignant image rejected by Stage 1 counts as a pipeline false negative.
- `results.json`, `summary.csv` and budget curves with mean/standard deviation across seeds.

Stage metrics include positive-class recall, specificity, macro-F1, ROC-AUC and FN/FP counts. Stage 1 positive class is tumor; Stage 2 positive class is malignant. Pipeline recall/FN are for malignant versus all other images, while pipeline accuracy/macro-F1 cover all three classes. Pipeline AUC uses the product of the two stage scores as an **uncalibrated ranking score**, not a clinical probability. Untyped tumor images cannot be included in three-class pipeline testing and their exclusion count is reported.

`--bootstrap 200` produces group-clustered 95% intervals for recall/AUC; replicates with only one class are excluded and their valid count is recorded. Across-seed standard deviation describes training/support-selection variability; it is not a patient-level confidence interval. Grouping by original image cannot establish patient-level independence.

For a smaller first real run:

```powershell
python run_experiments.py --manifest prepared/split/manifest.csv --data-root data --out runs/pilot --budgets 5 --seeds 42 --steps 100
```

Keep the final test locked. Do not repeatedly inspect its metrics to choose methods, hyperparameters or thresholds. Use validation for development, and record any prior exposure to the test data. Read the per-run support lists and metadata before interpreting label-efficiency claims.

## Demo

Set `app/config.yaml` to the presence and subtype artifacts from the same method/budget/seed and manifest. Then:

```powershell
streamlit run app/app.py
```

The app loads the saved prototypes and thresholds; it never rebuilds support from a directory that could contain validation/test images. Legacy `.pth` files are intentionally not accepted as protocol artifacts because they lack the necessary split/support/threshold provenance.

## Runnable verification

Run checks in a task-owned output directory:

```powershell
python test_protocol.py --out checks/regression
python run_experiments.py --manifest checks/regression/manifest.csv --data-root checks/regression --out checks/integration --budgets 2 --seeds 42 43 44 --steps 2 --val-every 1 --bootstrap 20 --image-size 32 --threads 2 --random-init
```

The checks cover grouped leakage, missing metadata, augmentation in test, annotation budgets, disjoint episodes, squared-distance scores, pipeline false negatives and save/load inference parity. The second command exercises all three methods, both stages, aggregation and error exports on synthetic images. **INTEGRATION_ONLY results demonstrate software behavior and must not be used as liver CT performance evidence.**

GitHub Actions runs the regression checks, a small synthetic training run for all three methods, and a Streamlit app check. It requires no hospital data or pre-existing checkpoints.

Historical notebooks remain available for reference. Use the commands above for the new protocol; preprocessing notebooks 01/02 should not be rerun on original data because their old in-place behavior does not preserve raw images.
