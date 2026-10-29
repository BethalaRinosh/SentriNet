# SentriNet
### Network Intrusion Detection Using Machine Learning
**Tagline:** *Every packet, judged.*

> **Project status: Steps 1–10 are implemented in the repository.** The author has run the Step 9 official evaluation and confirmed the full test suite passes (22 tests). Step 10 adds model persistence and reusable inference; the local artifact-training command and new Step 10 tests still need to be run after pulling the latest changes.

SentriNet is an academic micro project for the **Machine Learning Fundamentals** 3rd Internal Assessment at Karunya Institute of Technology and Sciences. It explores how classical machine-learning models can classify network connection records as normal or malicious and group attacks into five broad categories.

## Problem statement
Network intrusion detection systems help identify traffic patterns associated with malicious activity. This project studies a supervised-learning approach using the NSL-KDD benchmark, with both binary classification (Normal vs Attack) and five-class classification (Normal, DoS, Probe, R2L, U2R). It is an educational experiment, not a production firewall or a substitute for security monitoring.

## Dataset
**Primary dataset:** NSL-KDD, using the official `KDDTrain+` and `KDDTest+` split. We will not randomly re-split the official test set.

Required local files (already validated for this project):
- `KDDTrain+.txt` (full official training split)
- `KDDTest+.txt` (full official test split)

Put both files in `data/raw/`, so the final paths are `data/raw/KDDTrain+.txt` and `data/raw/KDDTest+.txt`. Use the full `KDDTrain+` and `KDDTest+` files, not the 20% subset or `KDDTest-21` subset. The University of New Brunswick's dataset page notes that its original dataset is no longer available, so use the public mirror at [jmnwong/NSL-KDD-Dataset](https://github.com/jmnwong/NSL-KDD-Dataset). Download the files directly from [KDDTrain+.txt](https://github.com/jmnwong/NSL-KDD-Dataset/blob/master/KDDTrain%2B.txt) and [KDDTest+.txt](https://github.com/jmnwong/NSL-KDD-Dataset/blob/master/KDDTest%2B.txt), using each file's **Download raw file** option. Raw data is intentionally excluded from version control. No dataset or synthetic substitute is bundled.

## Approach / planned pipeline
1. Inspect data shape, types, missing values, duplicates and labels.
2. Explore label distributions and relevant feature patterns with plots.
3. Preprocess categorical and numerical features inside a scikit-learn pipeline; fit transformations on training data only.
4. Investigate feature importance/selection using only the official training split.
5. Train at least five classical ML models for binary and five-class tasks.
6. Use stratified cross-validation and tune the two leading candidates on training data.
7. Evaluate against the fixed official test set with class-aware metrics, confusion matrices, ROC-AUC where applicable, and measured runtime.
8. Persist the Random Forest candidates selected by training-only cross-validation, including the fitted preprocessing pipelines and JSON metadata.
9. Load the saved pipelines for reproducible predictions; then expose inference through FastAPI and a plain HTML/CSS/JS interface.

Only steps explicitly marked as locally validated should be treated as verified. Step 10's source and tests are committed, but the new artifact-training run must still be validated locally.

## Dataset validation (Step 2)

The loader was run locally by the project author against the official NSL-KDD files. The captured terminal output reported:

| Split | Rows | Columns after target mapping | Missing values | Duplicate rows |
|---|---:|---:|---:|---:|
| KDDTrain+ | 125,973 | 45 | 0 | 0 |
| KDDTest+ | 22,544 | 45 | 0 | 0 |

The 45 columns are 41 connection features, the normalized raw attack label, difficulty metadata, and two derived targets (`binary_label` and `class_label`). Difficulty is retained for inspection but must not be used as a model input.

| Split | Normal | DoS | Probe | R2L | U2R |
|---|---:|---:|---:|---:|---:|
| Train | 67,343 | 45,927 | 11,656 | 995 | 52 |
| Test | 9,711 | 7,460 | 2,421 | 2,885 | 67 |

**Interpretation:** the five-class distribution changes substantially between the official splits, especially for R2L. U2R has very few records in both splits. This is a reason to report per-class recall and macro F1 rather than relying on accuracy alone. These are dataset counts, not model results.

## Exploratory data analysis (Step 3)

The script `src/eda.py` performs descriptive analysis on the **real official train and test files**. It does not synthesize records, fit preprocessing, rebalance the test set, or train a model.

Run from the repository root after Step 2's dataset files and dependencies are in place:

```bash
python -m src.eda
```

It prints class counts and percentages for both splits, protocol-type counts, and the largest train/test class-proportion differences. It also writes these five plots to `reports/eda/`:

- `class_counts_by_split.png`: raw five-class record counts.
- `class_proportions_by_split.png`: split-normalized percentages.
- `binary_counts_by_split.png`: Normal vs Attack counts.
- `protocol_counts_by_split.png`: protocol-type counts.
- `numeric_feature_distributions.png`: duration, source bytes, and destination bytes on a `log(1 + x)` scale.

The generated reports are ignored by Git; the script is the reproducible source of the figures. This avoids committing machine-generated plot files while keeping the raw dataset out of version control.

**Observations supported by the Step 2 dataset inspection:** the training split has 995 R2L and 52 U2R records, while the official test split has 2,885 R2L and 67 U2R records. R2L's proportion is much larger in the test split, so the official test distribution is not interchangeable with the training distribution. U2R has low support in both splits. The class-proportion plot is intended to make this gap easier to see; macro F1 and per-class recall will be important later. These are properties of the dataset, not evidence of model performance.

**Validation status:** confirmed locally. The EDA command completed and saved all five expected figures under `reports/eda/`. The observed test-minus-train class-proportion differences were R2L +12.01 percentage points, Normal -10.38 pp, DoS -3.37 pp, Probe +1.49 pp, and U2R +0.26 pp. These are descriptive dataset differences, not model metrics.

## Preprocessing pipeline (Step 4)

The module `src/preprocessing/pipeline.py` defines a reusable, **unfitted** preprocessing and model pipeline:

- Uses the 40 intended model inputs, excluding `num_outbound_cmds`, the constant-zero feature.
- One-hot encodes `protocol_type`, `service`, and `flag` with `handle_unknown="ignore"`, so an unseen category in the official test set does not crash inference.
- Standardizes numeric features with `StandardScaler`.
- Drops raw attack labels and difficulty metadata from model inputs. The target is selected explicitly as either `binary_label` or `class_label`.
- Keeps preprocessing inside the scikit-learn `Pipeline`, so fitting the pipeline on training data (or a cross-validation fold) learns scaler statistics and categories only from that training partition.
- Offers optional `class_weight="balanced"` for estimators that support it. The weights are computed by scikit-learn during each fit from that fit's labels. Estimators without `class_weight` must not enable this option.

Example use in later training code:

```python
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing.pipeline import build_model_pipeline, select_features_and_target

X_train, y_train = select_features_and_target(train_frame, target="class_label")
pipeline = build_model_pipeline(
    RandomForestClassifier(random_state=42),
    balance_classes=True,
)
pipeline.fit(X_train, y_train)
```

Do not fit on `KDDTest+`. Only transform/predict with the official test split after fitting the pipeline on training data. The code above is an example of the API, not a claim that a model has been trained.

Unit tests are in `tests/test_preprocessing.py`. Run them from the repository root:

```bash
python -m pytest tests/test_preprocessing.py -q
```

**Validation status:** the author ran `python -m pytest tests/test_preprocessing.py -q`; all four tests passed.

## Feature importance (Step 5)

`src/feature_selection.py` ranks the 40 original input features using Random Forest impurity-based importance. It loads **only** `data/raw/KDDTrain+.txt`; the official `KDDTest+.txt` is not loaded or used. By default, it draws a reproducible 30,000-row sample from the official training split (`random_state=42`) to keep the exploratory run practical. The one-hot encoded importances for `protocol_type`, `service`, and `flag` are summed back to their original feature groups, so the output is interpretable at the dataset-column level.

Run from the repository root:

```bash
python -m src.feature_selection
# Optional: inspect feature importance for the five-class target instead
python -m src.feature_selection --target class_label
```

The script writes `reports/feature_importance/binary_label_feature_importance.csv` and `reports/feature_importance/binary_label_top20_feature_importance.png` by default. These generated outputs are ignored by Git. Use `--sample-size` and `--n-estimators` to adjust runtime; both are reproducible with the fixed random state. Tests for aggregating one-hot importances are in `tests/test_feature_selection.py`.

**Interpretation caveat:** impurity-based importance is a useful exploratory ranking, not causal evidence, a held-out evaluation, or a guarantee that dropping a feature improves generalization. It can favor features with many possible split points. Final feature decisions should be checked against cross-validation and the untouched official test set during later steps.

**Validation status:** confirmed locally. The author ran the full test suite (`7 passed in 3.35s`) and `python -m src.feature_selection`; the CSV ranking and top-20 PNG were generated successfully.

### Observed feature-importance output

The locally run binary-target analysis ranked `src_bytes` (0.1373), `dst_bytes` (0.1126), `flag` (0.0914), and `service` (0.0699) as its top four features in the 30,000-row sample. These values are Random Forest impurity-importance scores aggregated to original feature names. They describe this exploratory fit only and are not final model metrics or causal claims.

## Binary baseline models (Step 6)

The module `src/training/binary_baselines.py` compares five classical classifiers for **Normal vs Attack**: Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, and Linear SVM.

Run from the repository root:

```bash
python -m src.training.binary_baselines
```

By default, the script samples 30,000 records reproducibly from the official `KDDTrain+.txt`, then makes an 80/20 stratified training/validation split. Each model's preprocessing is fitted only on its training subset. **It does not load `KDDTest+.txt`**; that official split remains reserved for final held-out evaluation in a later step. The sample is for practical baseline runtime, not a replacement dataset.

Generated outputs go to `reports/baseline_binary/` (ignored by Git):
- `binary_baseline_comparison.csv`: accuracy, macro precision/recall, macro and weighted F1, ROC-AUC where a continuous score is available, and fit/prediction times.
- `binary_baseline_details.json`: per-model confusion matrices and classification reports.
- `binary_baseline_macro_f1.png`: comparison chart sorted by macro F1.

Use `--sample-size` to change the reproducible sample size and `--validation-size` to change the held-out fraction from the sampled training records. The validation subset is not the official test set. These results are preliminary model-selection evidence; Step 8 will use stratified cross-validation and tuning, and the official test set will be used only during the final evaluation step.

**Validation status:** locally verified by the project author. The full test suite passed (`10 passed in 2.73s`), and all five baseline models completed on a reproducible 30,000-row sample from `KDDTrain+` with a stratified 24,000/6,000 training/validation split. The official `KDDTest+` was not loaded.

### Observed binary baseline results

These results are from the internal validation subset of `KDDTrain+`, not the official test set.

| Model | Accuracy | Macro F1 | Weighted F1 | ROC-AUC | Fit time (s) | Predict time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.998167 | 0.998158 | 0.998166 | 0.999773 | 0.5386 | 0.0405 |
| Decision Tree | 0.996500 | 0.996485 | 0.996500 | 0.996496 | 0.3128 | 0.0141 |
| K-Nearest Neighbors | 0.994333 | 0.994308 | 0.994333 | 0.998401 | 0.0629 | 2.0938 |
| Linear SVM | 0.974833 | 0.974710 | 0.974827 | 0.996313 | 0.6347 | 0.0165 |
| Logistic Regression | 0.973500 | 0.973352 | 0.973484 | 0.996490 | 0.5000 | 0.0159 |

**Interpretation:** Random Forest has the strongest macro F1 and ROC-AUC among these preliminary baselines. KNN fit quickly but had the slowest validation prediction time. These results are unusually strong and should be treated as internal validation evidence only: the random split comes from the same official training source and may not reflect the distribution shift in `KDDTest+`. We must not claim these are final generalization results. Later cross-validation/tuning and evaluation on the untouched official test split remain necessary.

## Five-class baseline models (Step 7)

The module `src/training/multiclass_baselines.py` compares five classical classifiers for `class_label`: **Normal, DoS, Probe, R2L, and U2R**. It reuses the existing preprocessing pipeline, so numeric scaling and categorical encoding are fitted on the internal training partition only.

Run from the repository root:

```bash
python -m src.training.multiclass_baselines
```

By default, the script uses all rows from the official `KDDTrain+.txt` and creates an 80/20 stratified internal training/validation split. To use a reproducible sample for a faster exploratory run, pass `--sample-size 30000`. The official `KDDTest+.txt` is not loaded. Models that support class weights use them to help address class imbalance; K-Nearest Neighbors does not support the same `class_weight` option.

Generated files go to `reports/baseline_multiclass/` (ignored by Git):
- `multiclass_baseline_comparison.csv`: accuracy, macro precision/recall, macro and weighted F1, and fit/prediction time.
- `multiclass_baseline_details.json`: confusion matrices and per-class classification reports for all five classes.
- `multiclass_baseline_macro_f1.png`: model comparison chart.

**Interpretation caution:** U2R has very few examples in the official training split, so its validation support is small and per-class metrics may be unstable. Macro F1 gives each class equal weight; weighted F1 reflects class support. Neither should be interpreted without the per-class report and confusion matrix. These are internal validation results, not official test results.

**Initial local run (before the solver adjustment below):** the full-data experiment completed on 125,973 rows, with 100,778 training and 25,195 validation records. Random Forest led macro F1 at 0.936115 (accuracy 0.998730, weighted F1 0.998709); Decision Tree reached macro F1 0.930726. U2R validation support was only 10 records, so its reported scores are highly unstable. The 30,000-row exploratory sample produced Random Forest macro F1 0.884361, with only three U2R validation records. These are internal validation results, not official test metrics.

The pasted test run had one failing assertion because the toy metric example has exactly one true record per class, making macro and weighted F1 equal. The test has been corrected to assert that expected equality. The subsequent full test run passed all 13 tests. The full-data experiment was rerun after configuring Linear SVM with `dual=False`, `max_iter=5000`, and `tol=1e-3`; the warning no longer appeared. This confirms the current baseline run completes cleanly, though it does not make the internal validation split a substitute for the official held-out test set.

## Stratified cross-validation and hyperparameter tuning (Step 8)

The module `src/training/cv_tuning.py` tunes the two strongest baseline candidates, **Random Forest** and **Decision Tree**, for both the binary target (`Normal` vs `Attack`) and five-class target (`Normal`, `DoS`, `Probe`, `R2L`, `U2R`).

Run from the repository root:

```bash
python -m src.training.cv_tuning
```

The default run uses all records in the official `KDDTrain+.txt`, five-fold shuffled `StratifiedKFold` with `random_state=42`, randomized search over four parameter combinations per model and task, and macro F1 as the selection metric. It searches four combinations of task/model (two models for each of two targets). Each fold fits its own preprocessing because scaling and one-hot encoding remain inside the scikit-learn pipeline. The search refits the winning pipeline on all supplied training rows in memory but does not save a model artifact; model persistence belongs to a later step.

Optional faster exploratory run:

```bash
python -m src.training.cv_tuning --sample-size 30000 --n-iter 2
```

Use `--cv-folds` to change the number of folds and `--n-iter` to change randomized candidates per model/task. The script checks that each class has at least as many rows as CV folds. The optional sample is drawn reproducibly from `KDDTrain+`; it is not a replacement for the full-data default.

Generated files are written to `reports/cv_tuning/` (ignored by Git):

- `cv_tuning_summary.csv`: best mean CV macro F1, fold-to-fold standard deviation, candidate count, and best parameter JSON for each task/model.
- `cv_tuning_details.json`: class counts, selected parameters, and per-candidate CV scores, variability, rank, and fit time.
- `cv_tuning_macro_f1.png`: comparison chart of the best mean CV macro F1.

**Evaluation boundary:** the official `KDDTest+.txt` is never loaded or scored in this step. Cross-validation results are model-selection evidence, not final held-out test performance. Rare U2R examples can make fold-level scores volatile even with stratification. Step 9 remains responsible for the final evaluation on the untouched official test split.

**Validation status:** verified locally. The author ran `python -m pytest -q` (17 passed in 2.52 seconds) and `python -m src.training.cv_tuning` on the full official training split. All four searches completed, all three report files were generated, and the command confirmed that `KDDTest+` was not loaded or scored.

## Official test evaluation (Step 9)

The module `src/training/official_evaluation.py` performs the final held-out evaluation for both binary (`Normal` vs `Attack`) and five-class (`Normal`, `DoS`, `Probe`, `R2L`, `U2R`) classification.

Run from the repository root after placing the official files in `data/raw/`:

```bash
python -m src.training.official_evaluation
```

**Leakage boundary:** the script first loads only `KDDTrain+`, then fits the four locked pipelines on the full training split. Only after all fits are complete does it load `KDDTest+` and score each fitted pipeline. Hyperparameters are fixed to the Step 8 best settings and are not selected using test results. Do not retune based on the official test metrics.

For each task, the script evaluates the tuned Random Forest and Decision Tree. It reports accuracy, macro and weighted precision/recall/F1, binary ROC-AUC from Attack probabilities, fit time, prediction time, throughput, fixed-label confusion matrices, and per-class classification reports. It also records train/test label counts, proportions, and test-minus-train percentage-point differences.

Outputs are saved locally under `reports/official_evaluation/` (ignored by Git):
- `official_evaluation_summary.csv`: one row per task/model with the aggregate metrics and timing.
- `official_evaluation_details.json`: per-class reports, confusion matrices, class support, and train/test label-distribution comparison.
- `official_evaluation_macro_f1.png`: macro-F1 comparison chart.
- `confusion-*.png`: one confusion-matrix image per task/model.

Tests for the locked parameters and metric calculations are in `tests/test_official_evaluation.py`. Run them with:

```bash
python -m pytest tests/test_official_evaluation.py -q
python -m pytest -q
```

**Local evaluation results (official `KDDTest+`, 22,544 rows):** the author ran the full-data evaluation successfully. After correcting a unit-test expectation for ROC-AUC, the author pulled the fix and confirmed the focused tests passed (5/5) and the full test suite passed (22/22), including a further clean rerun.

| Task | Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| Binary | Random Forest | 0.781139 | 0.818933 | 0.804454 | 0.780427 | 0.778695 | 0.957480 |
| Binary | Decision Tree | 0.793958 | 0.821881 | 0.814249 | 0.793723 | 0.792759 | 0.814087 |
| Five-class | Random Forest | 0.762110 | 0.731962 | 0.524459 | 0.514196 | 0.713959 | N/A |
| Five-class | Decision Tree | 0.750044 | 0.665830 | 0.518583 | 0.517832 | 0.709837 | N/A |

**Interpretation:** the Decision Tree narrowly leads the binary task on accuracy and macro F1, while Random Forest has substantially higher binary ROC-AUC, meaning its Attack scores rank examples well even though its default class threshold yields lower Attack recall. For five-class classification, both models struggle badly with R2L: Random Forest recall is 0.01 and Decision Tree recall is 0.05 on 2,885 R2L test records. The five-class Random Forest macro F1 (0.514) is much lower than its weighted F1 (0.714), showing that aggregate performance is much stronger on common classes than across all five classes equally.

The official test split has 2,885 R2L records compared with 995 in training. This distribution shift is a major limitation and likely contributes to poor R2L generalization. U2R test support is only 67, so its per-class scores are also unstable. These results are the held-out evaluation of the locked Step 8 configurations. **Do not tune hyperparameters against these test results**; further changes must be selected using training-only cross-validation or a separate validation split, with the official test retained as the final benchmark.

## Model persistence and reusable inference (Step 10)

The module `src/training/save_models.py` fits and saves the two Random Forest pipelines selected by Step 8's training-only cross-validation. Random Forest led the tested candidates on mean CV macro F1 for both targets: 0.998955 for binary classification and 0.946326 for five-class classification. These model choices are based on `KDDTrain+` CV results, not the official test scores.

After placing the official `KDDTrain+.txt` in `data/raw/`, run:

```bash
python -m src.training.save_models
```

The command loads only `KDDTrain+`, refits each selected pipeline on all 125,973 training rows, and writes these local artifacts under `models/`:

- `sentrinet_binary.joblib`: fitted preprocessing and binary Random Forest.
- `sentrinet_multiclass.joblib`: fitted preprocessing and five-class Random Forest.
- `metadata.json`: task, model parameters, feature order, label order, scikit-learn version, training-row count, and the training-only CV selection rationale.

The `models/` artifacts are ignored by Git and must be generated locally. Raw data and model binaries are not committed. Keep the generated metadata alongside the corresponding artifacts; joblib files should only be loaded from trusted sources.

### Inference helper

`src/inference/predict.py` loads a saved pipeline and predicts from a list of connection-feature dictionaries or a pandas DataFrame. Every record must contain the 40 model features in `src/preprocessing/pipeline.py`; extra fields are ignored and missing features produce a clear error. Unknown categorical values are handled by the fitted encoder's `handle_unknown="ignore"` setting.

Example after generating the artifacts:

```python
from src.inference.predict import predict_records

record = {
    # Supply all 40 features defined in MODEL_FEATURE_COLUMNS.
    # Example categorical values:
    "protocol_type": "tcp",
    "service": "http",
    "flag": "SF",
    # Add the remaining numeric and categorical features here.
}
result = predict_records([record], task="binary_label")
print(result)
```

Each result contains a predicted label and the maximum class score as `confidence`. Binary predictions also include `attack_probability`. These are model scores, not guaranteed calibrated probabilities. This helper does not capture live packets or replace a production network security control.

Run the focused tests and then the full suite:

```bash
python -m pytest tests/test_model_persistence.py -q
python -m pytest -q
```

**Validation status:** source files and tests are committed; the author must run the new tests and `python -m src.training.save_models` locally before this step can be marked fully validated.

## Results
Internal validation and cross-validation results are available for model selection. Official held-out test results should be taken only from the author's run of `reports/official_evaluation/official_evaluation_summary.csv`; they are not hard-coded into this README.

### Baseline validation results

| Task | Model | Accuracy | Macro F1 | Weighted F1 | Status |
|---|---|---:|---:|---:|---|
| Binary | Random Forest | 0.998167 | 0.998158 | 0.998166 | Internal validation only |
| Five-class | Random Forest | 0.998730 | 0.936115 | 0.998709 | Internal validation only |

### Five-fold cross-validation after randomized tuning

| Task | Model | Mean CV macro F1 | Fold standard deviation |
|---|---|---:|---:|
| Binary | Random Forest | 0.998955 | 0.000193 |
| Binary | Decision Tree | 0.998516 | 0.000199 |
| Five-class | Random Forest | 0.946326 | 0.020758 |
| Five-class | Decision Tree | 0.910170 | 0.019869 |

Selected parameters are recorded in `reports/cv_tuning/cv_tuning_summary.csv` and the detailed per-candidate records are in `reports/cv_tuning/cv_tuning_details.json`. The results favor Random Forest for both tasks under the tested search spaces. The five-class score varies more across folds, consistent with the difficulty and limited support of minority attack classes. The search tested only four parameter combinations per model/task, so these are good candidates, not proof of a globally optimal configuration.

Accuracy will not be reported alone. The final analysis will discuss class imbalance, rare R2L/U2R examples, and the distribution difference between NSL-KDD's official train and test sets.

## Screenshots / demo
- [Dashboard screenshot](reports/screenshots/dashboard.png) *(placeholder; not created yet)*
- [Live detection screenshot](reports/screenshots/live-detect.png) *(placeholder; not created yet)*
- [Demo GIF](reports/screenshots/demo.gif) *(placeholder; not created yet)*

Screenshots will be added only after the interface is implemented and run.

## Setup and run
**Current status:** the scaffold is not yet runnable as a complete application. The entry point and one-command startup instructions will be added with the backend implementation.

Recommended environment: Python 3.10 or newer.

```bash
git clone https://github.com/BethalaRinosh/SentriNet.git
cd SentriNet
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

With the two official files in `data/raw/`, validate the dataset from the repository root using:

```bash
python -m pip install -r requirements.txt
python -m src.inspect_dataset
```

This reports shapes, dtypes, missing values, duplicate rows, binary/five-class distributions, and raw attack-label counts for both official splits. It does not train a model. Dataset acquisition, notebook execution, training, API startup, and UI instructions will be documented in the relevant implementation steps.

## Planned folder structure
```text
SentriNet/
├── app/
│   ├── backend/
│   └── frontend/
├── data/
│   ├── raw/              # local dataset files; excluded from Git
│   └── processed/        # generated files; excluded from Git
├── models/               # generated model artifacts; excluded from Git
├── notebooks/
├── reports/              # generated reports; large artifacts excluded
├── src/
│   ├── preprocessing/
│   ├── feature_selection.py
│   ├── training/
│   │   ├── binary_baselines.py
│   │   └── multiclass_baselines.py
│   ├── evaluation/
│   └── utils/
├── tests/
├── .gitignore
├── requirements.txt
└── README.md
```

## Tech stack
- Python, pandas, NumPy
- scikit-learn and imbalanced-learn
- Matplotlib and Seaborn
- Jupyter Notebook
- FastAPI and Uvicorn
- Plain HTML, CSS and JavaScript
- Chart.js for browser charts
- pytest and joblib

The dependency list is an initial scaffold and will be validated as the modules are implemented.

## Limitations
- NSL-KDD is a benchmark dataset, not a complete representation of modern production network traffic.
- Some attack categories, particularly R2L and U2R, have very few samples and can be difficult to classify reliably.
- The official train/test distributions differ; test performance may therefore be lower than training or cross-validation performance.
- Classification results alone do not establish that a system is safe or effective in live deployments.
- Feature importance is exploratory and impurity-based. Step 9's official-test evaluation has been run locally and the full test suite passed (22 tests). Step 10's model-persistence and inference code still requires local validation. The FastAPI service and dashboard are not implemented yet.

## Future work
- Run Step 10's artifact-training command locally and confirm the generated model files and metadata; keep the official test set out of model selection.
- Build the FastAPI service and local interactive dashboard on top of the tested inference helper.
- Consider UNSW-NB15 or CICIDS2017 only as an optional extension after the NSL-KDD pipeline is working.
- Investigate explainability and robustness to distribution shift.

## Author
**Bethala Rinosh**  
Repository: https://github.com/BethalaRinosh/SentriNet

## Development log
- **Step 1:** Initial repository scaffold, dependency list, ignore rules and README skeleton.
- **Step 2:** Added the schema-validating NSL-KDD loader, standard binary and five-class target mapping, and an inspection command. The author ran the command locally: train = 125,973 rows, test = 22,544 rows; both had zero reported missing values and duplicate rows, and all labels mapped successfully.
- **Step 3:** Added and locally ran `src/eda.py`; all five figures were generated from the official files. The largest class proportion differences (test minus train) were R2L +12.01 pp and Normal -10.38 pp. No model training or test-set fitting occurred.
- **Step 4:** Added the 40-feature selector, one-hot encoding for the three categorical fields, numeric standardization, a scikit-learn model pipeline, optional balanced class weights, and preprocessing unit tests. The author ran the tests locally: 4 passed.
- **Step 5:** Added `src/feature_selection.py`, which fits a reproducible Random Forest on a sample of official `KDDTrain+` only, aggregates encoded importances to original features, and writes a CSV ranking and top-20 plot. The author ran the script successfully and the full test suite passed (7 tests).
- **Step 6:** Added five binary baseline classifiers, stratified validation from the official training split, class-aware metrics, timing, per-model JSON details, a macro-F1 comparison CSV/chart, and unit tests. The author ran the full suite locally (10 passed) and completed the baseline experiment on a 30,000-row training sample. Random Forest led the internal validation comparison (macro F1 0.998158); the official `KDDTest+` was not loaded. Results are documented as preliminary, not final test performance.
- **Step 7:** Added five-class baseline models for Normal, DoS, Probe, R2L, and U2R, per-class reports, confusion matrices, overall class-aware metrics, and reproducible report outputs. The author ran the full experiment on `KDDTrain+` only with a stratified 100,778/25,195 training/validation split; the full test suite passed (13 tests). After configuring Linear SVM with `dual=False`, `max_iter=5000`, and `tol=1e-3`, the rerun completed without a convergence warning. Random Forest led by macro F1 (0.936115), followed by Decision Tree (0.930726). `KDDTest+` was not loaded.
- **Step 8:** Added five-fold stratified cross-validation and randomized hyperparameter tuning for Random Forest and Decision Tree on both binary and five-class targets. The author ran the full test suite (17 passed in 2.52 seconds) and the full-data tuning run successfully. Best mean CV macro F1 was 0.998955 for binary Random Forest and 0.946326 for five-class Random Forest. Generated CSV, JSON, and PNG reports were confirmed; `KDDTest+` was not loaded or scored.
- **Step 9:** Added `src/training/official_evaluation.py` and `tests/test_official_evaluation.py`. The script locks the Step 8 parameters, fits binary and five-class Random Forest/Decision Tree pipelines on full `KDDTrain+`, then loads `KDDTest+` for held-out evaluation. It writes aggregate metrics, detailed per-class reports, confusion matrices, runtime/throughput, and train/test distribution comparisons. The author ran the official evaluation and confirmed the focused tests (5 passed) and full suite (22 passed), including a clean rerun.
- **Step 10:** Added `src/training/save_models.py` to refit the two Random Forest candidates selected by training-only CV and save their complete preprocessing/model pipelines with provenance metadata. Added `src/inference/predict.py` for validated, repeatable prediction from feature records and `tests/test_model_persistence.py` for metadata and inference behavior. The README was corrected to reflect Step 9's actual validation status. Step 10's tests and artifact-training command are pending local execution.
