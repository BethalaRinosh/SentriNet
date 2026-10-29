# SentriNet
### Network Intrusion Detection Using Machine Learning
**Tagline:** *Every packet, judged.*

> **Project status: Step 2 of 16 (blocked: dataset needed locally).** The repository scaffold is committed. Dataset loading and label mapping have not started because the required NSL-KDD files could not be downloaded into this execution environment. No models have been trained and no metrics are available. We will not fabricate or synthesize data.

SentriNet is an academic micro project for the **Machine Learning Fundamentals** 3rd Internal Assessment at Karunya Institute of Technology and Sciences. It explores how classical machine-learning models can classify network connection records as normal or malicious and group attacks into five broad categories.

## Problem statement
Network intrusion detection systems help identify traffic patterns associated with malicious activity. This project studies a supervised-learning approach using the NSL-KDD benchmark, with both binary classification (Normal vs Attack) and five-class classification (Normal, DoS, Probe, R2L, U2R). It is an educational experiment, not a production firewall or a substitute for security monitoring.

## Dataset
**Primary dataset:** NSL-KDD, using the official `KDDTrain+` and `KDDTest+` split. We will not randomly re-split the official test set.

Required files (download these before resuming Step 2):
- `KDDTrain+.txt` (full official training split)
- `KDDTest+.txt` (full official test split)

Put both files in `data/raw/`, so the final paths are `data/raw/KDDTrain+.txt` and `data/raw/KDDTest+.txt`. Use the full `KDDTrain+` and `KDDTest+` files, not the 20% subset or `KDDTest-21` subset. The University of New Brunswick's dataset page notes that its original dataset is no longer available, so use the public mirror at [jmnwong/NSL-KDD-Dataset](https://github.com/jmnwong/NSL-KDD-Dataset). Download the files directly from [KDDTrain+.txt](https://github.com/jmnwong/NSL-KDD-Dataset/blob/master/KDDTrain%2B.txt) and [KDDTest+.txt](https://github.com/jmnwong/NSL-KDD-Dataset/blob/master/KDDTest%2B.txt), using each file's **Download raw file** option. Raw data is intentionally excluded from version control. No dataset or synthetic substitute is bundled.

## Approach / planned pipeline
1. Inspect data shape, types, missing values, duplicates and labels.
2. Explore label distributions and relevant feature patterns with plots.
3. Preprocess categorical and numerical features inside a scikit-learn pipeline; fit transformations on training data only.
4. Investigate feature importance/selection.
5. Train at least five classical ML models for binary and five-class tasks.
6. Use stratified cross-validation and tune the two leading candidates on training data.
7. Evaluate against the fixed official test set with class-aware metrics, confusion matrices, ROC-AUC where applicable, and measured runtime.
8. Save the chosen fitted pipeline and serve predictions through a FastAPI backend with a plain HTML/CSS/JS interface.

These are planned steps, not claims that the pipeline is implemented yet.

## Results
**No measured results yet.** This table will be populated from the evaluation scripts after training and testing have run successfully.

| Task | Model | Accuracy | Macro F1 | Weighted F1 | Status |
|---|---|---:|---:|---:|---|
| Binary | Not trained | N/A | N/A | N/A | Pending dataset and experiments |
| Five-class | Not trained | N/A | N/A | N/A | Pending dataset and experiments |

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

Dataset acquisition, notebook execution, training, API startup, and UI instructions will be documented in the relevant implementation steps. Do not attempt to train until the official NSL-KDD files are present.

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
│   ├── training/
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
- At this initial stage, no preprocessing, models, metrics, or UI features have been implemented.

## Future work
- Complete the leakage-safe binary and five-class ML experiments.
- Add a reproducible model comparison and a local interactive dashboard.
- Consider UNSW-NB15 or CICIDS2017 only as an optional extension after the NSL-KDD pipeline is working.
- Investigate explainability and robustness to distribution shift.

## Author
**Bethala Rinosh**  
Repository: https://github.com/BethalaRinosh/SentriNet

## Development log
- **Step 1:** Initial repository scaffold, dependency list, ignore rules and README skeleton.
- **Step 2 (blocked):** Confirmed the required full NSL-KDD files and documented a public mirror. Dataset loading and label mapping are waiting for the files to be placed in `data/raw/`; no data was fabricated.
