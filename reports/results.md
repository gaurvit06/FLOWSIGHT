# Cross-validated results

Stratified 5-fold CV on 200,000 training rows (held-out test set not used). Mean ± std over folds.

|     | macro_f1        | min_attack_recall   | fpr             |
|:----|:----------------|:--------------------|:----------------|
| rf  | 0.9962 ± 0.0017 | 0.9908 ± 0.0039     | 0.0008 ± 0.0004 |
| xgb | 0.9960 ± 0.0020 | 0.9912 ± 0.0043     | 0.0010 ± 0.0005 |
| svm | 0.6667 ± 0.0074 | 0.0000 ± 0.0000     | 0.0188 ± 0.0004 |
