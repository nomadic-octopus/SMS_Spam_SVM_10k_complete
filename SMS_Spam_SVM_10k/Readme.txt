SMS SPAM DETECTION — IMPORTANT CHANGES

Previous project:
- Used ~5,572 SMS examples, 747 spam and 4,825 ham before deduplication.
- Oversampled real spam examples by duplicating them in the training partition.
- Older notebook result comparisons no longer apply to this rebuilt dataset.

Updated project:
- 10,000 distinct SMS messages: 5,000 spam and 5,000 ham.
- Original (deduplicated): 642 spam + 4,515 ham.
- Added synthetic (template-generated): 4,358 spam + 485 ham.
- Every synthetic SMS is explicitly marked in spam.csv, and can be regenerated.
- New training includes synthetic records, but final testing uses only held-out
  original real-world SMS (1,032 messages).
- Group-aware cross-validation reduces leakage between variations of the same
  synthetic template; TF-IDF and LinearSVC are fitted together.

On the held-out real SMS, the updated SVM achieved:
- Accuracy 99.22% ; Spam precision 98.39%
- Spam recall 95.31% ; Spam F1 96.83%

The original-only baseline is in SMS_Spam_SVM_old.ipynb.
On the same holdout it achieved accuracy 98.93%, spam precision 96.06%,
spam recall 95.31%, and spam F1 95.69%.
Synthetic data can make training less representative of real-world spam;
do not claim that all 10,000 SMS examples were collected from real users.

VS Code quick run (after installing requirements):
.venv\Scripts\python.exe train.py
.venv\Scripts\python.exe predict.py "Free money! Claim now"
