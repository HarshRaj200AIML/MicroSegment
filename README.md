# MicroSegment

**A Streamlit research workbench for district level MSME registration analysis and unsupervised clustering.**

MicroSegment explores tabular data, supports preprocessing and clustering, and reports descriptive profiles with provenance-aware outputs. The supplied MSME registration data is aggregated by district. Results describe district records and must not be presented as segmentation of individual entrepreneurs.

> Clusters summarize similarity in the selected data. They are not verified measures of business success, creditworthiness, or causal impact.

## What it includes

- CSV, XLS, XLSX, and Stata DTA loading, including Stata labels when supported by `pyreadstat`.
- Dataset exploration, missingness and duplicate checks, column role detection, preprocessing, and feature selection.
- K-Means and DBSCAN clustering, with optional K-Medoids when `scikit-learn-extra` is installed.
- Elbow, silhouette, Davies-Bouldin, Calinski-Harabasz, stability, PCA, cluster profiles, and model evaluation where the data supports them.
- New record assignments, provenance metadata, and CSV, JSON, ZIP, and report downloads.
- A responsive dark-mode Streamlit interface with shared dataset and model state across the analysis pages.

Metrics and visualizations are calculated from the active dataset and fitted model. Undefined metrics are reported as unavailable. PCA is a visualization projection and does not establish that the plotted dimensions preserve all of the data's structure.

## Repository layout

```text
MicroSegment/
├── app.py                 # Streamlit dashboard entrypoint
├── pages/                 # Dataset, preprocessing, analytics, clustering, and reports
├── src/                   # Data loading, modeling, metrics, provenance, and UI
├── assets/style.css       # Shared visual design
├── .streamlit/config.toml # Streamlit theme
├── data/raw/              # Local input files; source data is not committed
├── notebooks/             # Reproducible analysis notebooks
├── tests/                 # Automated project tests
├── requirements.txt       # Pinned app runtime dependencies
└── requirements-dev.txt   # Notebook and test dependencies
```

Datasets, trained model files, generated outputs, local environments, and design backups are excluded from Git. The deployed app starts without the local dataset or saved model; upload a permitted dataset in Dataset Explorer and fit a model to produce results.

## Run locally

Python 3.14 is the deployment target. From the repository root:

```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m streamlit run app.py
```

For a runtime-only environment, install `requirements.txt`. To run the automated test suite locally:

```powershell
python -m pytest
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub. The source repository can remain private.
2. Connect the GitHub repository in [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select the `main` branch and `app.py` as the entrypoint.
4. In Advanced settings, select Python 3.14. Runtime dependencies are pinned in `requirements.txt`.

When deployed from this private repository, the app is private to developers in the Streamlit workspace by default. You can invite viewers or make it public from the app's sharing settings. Uploaded files are processed by the hosted application, so use only data you are authorized to process and share; do not upload personal or confidential records. Treat files written by the hosted app as temporary and download exports you need to keep.

## Data handling and methodology

- Add permitted local source files to `data/raw/`, or upload them through Dataset Explorer. The `.gitignore` rules keep source data out of GitHub.
- The project workspace currently uses district-level MSME registration aggregates, not individual entrepreneur microdata. Clusters describe district registration records only.
- Numeric columns use median imputation and standardization; categorical columns use mode imputation and one-hot encoding. Identifier-like columns are excluded from default feature candidates.
- DBSCAN noise is represented by label `-1` and handled separately in cluster summaries and applicable metrics.
- Model restoration is validated against the active dataset and saved assignment labels. A saved artifact without sufficient provenance is not represented as a verified run.
- K-Medoids is optional and requires `scikit-learn-extra`; it is not part of the standard app environment.

## Troubleshooting

- **No dataset is loaded:** upload a supported file in Dataset Explorer or place a permitted file in `data/raw/` for local use.
- **Stata import issues:** confirm `pyreadstat` is installed; the loader has a pandas fallback, but label metadata may be unavailable.
- **Legacy Excel `.xls` import issues:** confirm `xlrd` is installed. `.xlsx` files use `openpyxl`.
- **Unavailable metric:** one cluster, all-noise DBSCAN results, or too few usable rows can make an internal score undefined.
- **Large datasets:** some evaluation and visualization steps can use substantial memory. Check the dataset size and available resources before fitting.
