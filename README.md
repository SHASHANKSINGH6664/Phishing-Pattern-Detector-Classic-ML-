# Phishing Pattern Detector

Detects phishing URLs using machine learning. Trained on 11,430 URLs (50/50 split).

## Results
| Metric | Score |
|--------|-------|
| Accuracy | **93.09%** |
| ROC-AUC | **0.9826** |
| F1-Score | 0.93 |

## How It Works
Combines two feature branches:
- **TF-IDF char n-grams** — learns suspicious character patterns from raw URLs
- **26 hand-crafted features** — URL length, entropy, subdomain count, risky TLDs (`.tk`, `.xyz`), suspicious keywords (`login`, `verify`, `secure`), IP in host, HTTPS check, brand impersonation signals

**Model:** VotingClassifier (RandomForest + GradientBoosting)

## Setup
```bash
pip install scikit-learn pandas numpy
python phishing_pattern_detector.py
```

## Dataset
CSV with two columns: `url`, `status` (`phishing` / `legitimate`)  
Path set in `DATA_PATH` at the top of the script.

