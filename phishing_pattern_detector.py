"""
============================================================
  Phishing Pattern Detector
  Dataset  : data.csv  (url, status)
  Author   : Generated for Shashank Shekhar
  Level    : Medium  —  Feature Engineering + Ensemble Model
============================================================

Features extracted from URL:
  Lexical  : length, digit/letter ratio, special-char counts, entropy
  Structural: dots, hyphens, slashes, subdomains, path depth
  Token    : presence of suspicious keywords, brand impersonation signals
  TLD / IP : numeric IP in host, TLD risk, port presence
  HTTPS    : scheme check

Model Pipeline:
  TF-IDF char n-grams (2-4) on raw URL
  + Hand-crafted numerical features
  → RandomForest  (primary)
  → GradientBoosting (comparison)
  → VotingClassifier (final)
"""

# ── Imports ─────────────────────────────────────────────────────────────────
import re
import math
import warnings
import numpy as np
import pandas as pd

from urllib.parse import urlparse

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH = r"C:\Users\SHASHANK SHEKHAR\Desktop\PPPPPPROJECT\data.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "sign-in", "verify", "update", "secure", "account",
    "banking", "payment", "confirm", "password", "credential", "webscr",
    "paypal", "apple", "amazon", "google", "microsoft", "support",
    "validation", "invoice", "security", "alert", "suspended", "unusual",
]

RISKY_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top",
    ".club", ".online", ".site", ".work", ".party",
}


# ── Feature Engineering ──────────────────────────────────────────────────────
def shannon_entropy(s: str) -> float:
    """Calculate Shannon entropy of a string."""
    if not s:
        return 0.0
    prob = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in prob)


def extract_features(url: str) -> dict:
    """Return a dict of numerical features for a single URL."""
    try:
        parsed = urlparse(url if url.startswith("http") else "http://" + url)
        host = parsed.netloc or ""
        path = parsed.path or ""
        full = url
    except Exception:
        host, path, full = "", "", url

    # ── Length features ──────────────────────────────────────────────────
    f = {}
    f["url_length"]  = len(full)
    f["host_length"] = len(host)
    f["path_length"] = len(path)

    # ── Character-level features ─────────────────────────────────────────
    f["num_dots"]      = full.count(".")
    f["num_hyphens"]   = full.count("-")
    f["num_slashes"]   = full.count("/")
    f["num_at"]        = full.count("@")
    f["num_equals"]    = full.count("=")
    f["num_ampersand"] = full.count("&")
    f["num_percent"]   = full.count("%")
    f["num_digits"]    = sum(c.isdigit() for c in full)
    f["digit_ratio"]   = f["num_digits"] / max(len(full), 1)
    f["letter_ratio"]  = sum(c.isalpha() for c in full) / max(len(full), 1)

    # ── Entropy ──────────────────────────────────────────────────────────
    f["url_entropy"]  = shannon_entropy(full)
    f["host_entropy"] = shannon_entropy(host)

    # ── Host / subdomain features ────────────────────────────────────────
    host_clean = host.split(":")[0]          # strip port
    host_parts = host_clean.split(".")
    f["num_subdomains"]  = max(0, len(host_parts) - 2)
    f["host_num_hyphens"] = host_clean.count("-")
    f["host_num_digits"]  = sum(c.isdigit() for c in host_clean)
    f["has_port"]         = int(":" in host)

    # ── IP address in host ───────────────────────────────────────────────
    ip_pattern = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
    f["has_ip"] = int(bool(ip_pattern.match(host_clean)))

    # ── Scheme ───────────────────────────────────────────────────────────
    f["is_https"] = int(full.lower().startswith("https"))

    # ── Path depth ───────────────────────────────────────────────────────
    f["path_depth"] = len([p for p in path.split("/") if p])

    # ── TLD risk ─────────────────────────────────────────────────────────
    tld = "." + host_clean.rsplit(".", 1)[-1] if "." in host_clean else ""
    f["risky_tld"] = int(tld.lower() in RISKY_TLDS)

    # ── Suspicious keywords ──────────────────────────────────────────────
    url_lower = full.lower()
    f["num_suspicious_kw"] = sum(kw in url_lower for kw in SUSPICIOUS_KEYWORDS)
    f["has_suspicious_kw"] = int(f["num_suspicious_kw"] > 0)

    # ── Brand name in subdomain (impersonation signal) ───────────────────
    brands = ["paypal", "apple", "amazon", "google", "microsoft", "facebook",
              "netflix", "dropbox", "linkedin"]
    f["brand_in_subdomain"] = int(
        any(b in ".".join(host_parts[:-2]).lower() for b in brands)
        if len(host_parts) > 2 else False
    )

    # ── Query string length ──────────────────────────────────────────────
    qs = parsed.query or ""
    f["query_length"]    = len(qs)
    f["num_query_params"] = qs.count("&") + (1 if qs else 0)

    return f


# ── Custom sklearn Transformers ──────────────────────────────────────────────
class NumericalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Applies extract_features() to each URL; returns a numpy array."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = [extract_features(url) for url in X]
        df = pd.DataFrame(rows).fillna(0)
        return df.values.astype(np.float32)


class URLSelector(BaseEstimator, TransformerMixin):
    """Passes raw URL strings through (for TF-IDF branch)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return list(X)


# ── Build Pipeline ───────────────────────────────────────────────────────────
def build_pipeline():
    tfidf_branch = Pipeline([
        ("selector", URLSelector()),
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=8000,
            sublinear_tf=True,
        )),
    ])

    numerical_branch = Pipeline([
        ("features", NumericalFeatureExtractor()),
        ("scaler",   StandardScaler()),
    ])

    combined_features = FeatureUnion([
        ("tfidf_chars", tfidf_branch),
        ("hand_crafted", numerical_branch),
    ])

    rf  = RandomForestClassifier(n_estimators=200, max_depth=None,
                                  min_samples_leaf=2, n_jobs=-1,
                                  random_state=RANDOM_STATE)
    gb  = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                      max_depth=5, random_state=RANDOM_STATE)

    voting = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        voting="soft",
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("features", combined_features),
        ("clf",      voting),
    ])
    return pipeline


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("       PHISHING PATTERN DETECTOR")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────
    print(f"\n[1] Loading dataset from:\n    {DATA_PATH}")
    df = pd.read_csv("data.csv")
    print(f"    Rows: {len(df):,}  |  Columns: {list(df.columns)}")

    # normalise label column
    df.columns = [c.strip().lower() for c in df.columns]
    df["status"] = df["status"].str.strip().str.lower()
    df.dropna(subset=["url", "status"], inplace=True)

    X = np.array(df["url"].astype(str).tolist())
    y = np.array((df["status"] == "phishing").astype(int).tolist())

    print(f"    Phishing : {y.sum():,}  |  Legitimate : {(1-y).sum():,}")

    # ── Train / test split ───────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[2] Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

    # ── Build & train ────────────────────────────────────────────────────
    print("\n[3] Building pipeline (TF-IDF char n-grams + hand-crafted features)")
    print("    Classifier: VotingClassifier (RandomForest + GradientBoosting)")
    pipeline = build_pipeline()

    print("\n[4] Training … (this may take ~1-2 min)")
    pipeline.fit(X_train, y_train)
    print("    Training complete.")

    # ── Evaluation ───────────────────────────────────────────────────────
    print("\n[5] Evaluating on held-out test set …")
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\n" + "─" * 50)
    print(f"  Accuracy  : {acc * 100:.2f}%")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print("─" * 50)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["Legitimate", "Phishing"]))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

    # ── 5-fold Cross-Validation on FULL dataset ──────────────────────────
    print("\n[6] 5-Fold Stratified Cross-Validation on full dataset …")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv,
                                 scoring="accuracy", n_jobs=-1)
    print(f"  CV Accuracy: {cv_scores.mean()*100:.2f}% "
          f"(± {cv_scores.std()*100:.2f}%)")
    print(f"  Per-fold   : {[f'{s*100:.2f}%' for s in cv_scores]}")

    # ── Live demo prediction ─────────────────────────────────────────────
    print("\n[7] Quick demo — predict a few URLs:")
    demo_urls = [
        "https://www.google.com",
        "http://paypal.com-secure-login.tk/update",
        "https://github.com/openai/gpt-4",
        "http://192.168.1.1/login.php?user=admin",
        "https://support-appleld.com.secureupdate.duilawyeryork.com/ap/update",
        "https://www.amazon.com/dp/B09G9HD6PD",
    ]
    preds  = pipeline.predict(demo_urls)
    probas = pipeline.predict_proba(demo_urls)[:, 1]

    label_map = {0: "LEGITIMATE", 1: "PHISHING  "}
    print(f"\n  {'URL':<60}  {'Prediction':<12}  Phishing%")
    print("  " + "-" * 80)
    for url, pred, prob in zip(demo_urls, preds, probas):
        short = (url[:57] + "...") if len(url) > 60 else url
        print(f"  {short:<60}  {label_map[pred]}  {prob*100:6.1f}%")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)

    return pipeline   # return trained model for re-use


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = main()
