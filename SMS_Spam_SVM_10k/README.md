# SMS Spam Detection — 10,000 messages (SVM)

This is the updated version of your SMS Spam Detection project for **VS Code on Windows**.
It is an end-to-end SMS classifier using **TF-IDF + Support Vector Machine (LinearSVC)**.
The project was tested with Python 3.13 and scikit-learn 1.8.0 (Python 3.11–3.13 recommended).

## What's in the dataset?

`spam.csv` contains **exactly 10,000 unique text messages**, with **5,000 spam** and **5,000 ham**:

| Class | Original public SMS | Newly generated synthetic SMS | Total |
| --- | ---: | ---: | ---: |
| Spam | 642 | 4,358 | 5,000 |
| Ham | 4,515 | 485 | 5,000 |
| **Total** | **5,157** | **4,843** | **10,000** |

Your supplied `SMS.csv` is kept unchanged as the **original source** (5,572 rows before deduplication).
The extra examples are **synthetic, produced locally from written templates**, not newly collected genuine SMS. **Do not describe all 10,000 as real observations.**
Synthetic examples can help with code testing and augmenting training, but repetitive template structure may overstate model generalization. For research-quality benchmarking, use a larger independently collected real-world SMS dataset.

The `spam.csv` columns are:
- `v1`: `ham` or `spam` (compatible with your original notebooks)
- `v2`: complete SMS message
- `source`: `original` or `synthetic`
- `template_family`: template ID for synthetic messages; per-text ID for original messages (used for leakage-resistant CV)

## Files and what they do

| File | Function |
| --- | --- |
| `spam.csv` | New 10,000-row labeled dataset |
| `SMS.csv` | Your unchanged original dataset; used to rebuild `spam.csv` |
| `build_dataset.py` | Recreates the balanced 10,000-row CSV deterministically |
| `download_data.py` | Rebuilds `spam.csv` from local `SMS.csv`; downloads original only if missing |
| `train.py` | Validates CSV, performs grouped cross-validation, trains SVM and tests on real SMS |
| `predict.py` | Command-line and interactive classification |
| `sms_spam_pipeline.pkl` | Pretrained TF-IDF + LinearSVC model (from new CSV) |
| `model_info.json` | Dataset checksum, installed sklearn version, training/evaluation details |
| `SMS_Spam_SVM.ipynb` | Main updated exploration, training, and evaluation notebook |
| `SMS_Spam_SVM_new.ipynb` | Updated augmented-training notebook (same approach as `train.py`) |
| `SMS_Spam_SVM_old.ipynb` | Updated **original-real-only baseline** notebook (comparison) |
| `requirements.txt` | Python libraries |
| `Readme.txt` | Concise comparison and interpretation |
| `test_system.py` | Offline checks for dataset, holdout split, model and two predictions |

## Step-by-step installation in VS Code (Windows)

1. Install **Python 3.11 or newer**, **VS Code**, and the official **Python** and **Jupyter** extensions in VS Code.
2. Unzip the supplied project zip and open the **`SMS_Spam_SVM_10k`** folder: VS Code → **File → Open Folder**.
3. Open **Terminal → New Terminal**. From inside that folder, run:

   ```powershell
   py -m venv .venv
   .venv\Scripts\python.exe -m pip install --upgrade pip
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

   This direct interpreter path works even when PowerShell blocks activation scripts.

4. Train the updated model:

   ```powershell
   .venv\Scripts\python.exe train.py
   ```

   This rebuilds `sms_spam_pipeline.pkl` and `model_info.json` using your local Python environment.

5. Predict one message:

   ```powershell
   .venv\Scripts\python.exe predict.py "Congratulations! You won a free gift card. Claim it now."
   ```

6. Enter messages interactively:

   ```powershell
   .venv\Scripts\python.exe predict.py
   ```

   Type a message, press Enter, and use `exit` to close the predictor.

7. Optionally run a quick project validation:

   ```powershell
   .venv\Scripts\python.exe test_system.py
   ```

8. To view the notebooks, open `SMS_Spam_SVM.ipynb`, click **Select Kernel** (top right) → **Python Environments** → the project's `.venv`, then **Run All**. Open the **project folder itself** in VS Code so local imports work.

> If Windows doesn't recognize `py`, use `python` instead and check `python --version`.

## How training and evaluation work

1. Read `spam.csv` and check 5,000 ham + 5,000 spam; reject missing/duplicate rows.
2. Separate **original** and **synthetic** records using `source`.
3. Stratify-split **only the original SMS** into 80% train / 20% held-out test.
4. Add **all synthetic messages to the training partition only**. No synthetic examples enter the final real-message test.
5. Fit `TfidfVectorizer(ngram_range=(1,2), max_features=12000)` and `LinearSVC(class_weight='balanced')` in a single scikit-learn pipeline.
6. Tune C = 0.1, 1.0, 10.0 with 5-fold **StratifiedGroupKFold**, using template IDs as groups to keep synthetic siblings in the same CV fold.
7. Report accuracy, spam precision, spam recall, F1, classification report and confusion matrix **on held-out original SMS**.
8. Save the trained pipeline and model metadata. `predict.py` checks the dataset hash and sklearn version to avoid loading an outdated model; it can retrain automatically.

### Results obtained during project preparation

Using `spam.csv` in this deliverable and scikit-learn 1.8.0:

| Measure (held-out original SMS only) | Result |
| --- | ---: |
| Original SMS in final test | 1,032 (904 ham, 128 spam) |
| Accuracy | **99.22%** |
| Spam precision | **98.39%** |
| Spam recall | **95.31%** |
| Spam F1 | **96.83%** |
| Correct ham messages | 902 of 904 |
| Correct spam messages | 122 of 128 |
| Best SVM C | 1.0 |

For comparison, the **original-only baseline** (`SMS_Spam_SVM_old.ipynb`) scored **98.93% accuracy**, **96.06% spam precision**, **95.31% spam recall**, and **95.69% spam F1** on the *same* held-out original SMS test set. The augmented model improved precision and F1 in this test, with unchanged recall; these results are specific to this data and evaluation design.

Scores may differ under a changed dataset, split or scikit-learn release. The test uses genuine SMS, but these results **do not prove 99% performance on new contemporary SMS**. Some originally collected SMS are old or similar to each other, and synthetic training messages have recognizable patterns. A separate recent real-world test collection is strongly recommended for any formal evaluation.

## Recreate your dataset

The project ships with `spam.csv`, so rebuilding is optional:

```powershell
.venv\Scripts\python.exe download_data.py --force
.venv\Scripts\python.exe train.py
```

`download_data.py` will use the bundled `SMS.csv` offline. If `SMS.csv` is also missing, the script attempts to obtain the public SMS collection from the upstream tutorial URL (internet required).

## Common problems

- **`ModuleNotFoundError: No module named 'sklearn'`**: install packages using the exact `.venv\Scripts\python.exe -m pip install -r requirements.txt` command.
- **`No such file or directory: spam.csv`**: keep the downloaded folder together; run `download_data.py --force` if necessary.
- **Model created in an older environment**: rerun `train.py`; a pickled sklearn model is most portable with the same library version.
- **Notebook can't import `train`**: in VS Code, open the whole project folder and select the project interpreter as the notebook kernel.
- **Wrong training dataset after modifying the CSV**: just rerun `train.py`; `predict.py` also checks the CSV checksum before predictions.
- **PowerShell won't activate virtualenv**: the commands above don't require activating it.

## Dataset provenance

The original data is the SMS Spam Collection (Almeida et al.), supplied in `SMS.csv` and often distributed as the UCI / Kaggle SMS Spam Collection:

- https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

The additional examples in this project were generated from the explicitly documented templates in `build_dataset.py`, not downloaded from UCI or Kaggle.
