"""
analytics/segmentation.py
--------------------------
KMeans-based customer segmentation for the Churn Intelligence Platform.

Segments
--------
Four segments are derived from clustering on behavioural & risk features:

  Loyal     — high tenure, low churn probability, moderate-to-high charges
  Premium   — high charges, high service count, lower churn risk
  Risky     — high churn probability, short tenure, month-to-month contract
  Inactive  — high inactivity days, low engagement, elevated churn risk

The mapping from cluster centroids → human labels is done by scoring each
centroid against a rubric (see ``_label_clusters``).

Public API
----------
- ``CustomerSegmenter``                        (class)
  - ``.fit(df)``                        → fitted segmenter
  - ``.predict(df)`` → df with segment columns appended
  - ``.segment_profile()``              → {label: {stat: value}}
  - ``.save() / .load()``               → joblib persistence
- ``segment_customers(df)``             convenience wrapper
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ml.config import MODEL_DIR, N_CLUSTERS, RANDOM_STATE

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_SEGMENTER_FILENAME = "customer_segmenter.joblib"

# Features used for clustering (must exist in the input DataFrame after prediction)
_CLUSTER_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "InactiveDays",
    "churn_probability",  # added by predictor
    "service_count",      # added by feature engineering
    "engagement_score",   # added by feature engineering
]

# Segment labels in priority order (used by _label_clusters rubric)
SEGMENT_LABELS = ["Loyal", "Premium", "Risky", "Inactive"]

# ── Segment rubric ────────────────────────────────────────────────────────────

def _score_centroid(centroid: dict[str, float]) -> str:
    """
    Map a cluster centroid to a human-readable segment label using
    a simple priority-based rubric.

    Parameters
    ----------
    centroid : dict of {feature_name: mean_value} for the cluster

    Returns
    -------
    str : one of "Loyal", "Premium", "Risky", "Inactive"
    """
    tenure      = centroid.get("tenure", 0)
    charges     = centroid.get("MonthlyCharges", 0)
    inactive    = centroid.get("InactiveDays", 0)
    churn_prob  = centroid.get("churn_probability", 0)
    engagement  = centroid.get("engagement_score", 0)

    # Rules applied in priority order — first match wins
    if churn_prob >= 0.55 and inactive >= 40:
        return "Inactive"
    if churn_prob >= 0.50 and tenure <= 18:
        return "Risky"
    if charges >= 70 and engagement >= 40:
        return "Premium"
    # Default: loyal (long tenure, low risk)
    return "Loyal"


class CustomerSegmenter:
    """
    KMeans customer segmenter with centroid-to-label mapping.

    Usage
    -----
    ::
        segmenter = CustomerSegmenter()
        segmenter.fit(df_with_predictions)
        df_segmented = segmenter.predict(df_with_predictions)
        profile = segmenter.segment_profile(df_segmented)
    """

    def __init__(self, n_clusters: int = N_CLUSTERS) -> None:
        self.n_clusters  = n_clusters
        self._kmeans     = KMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
            n_init=20,
            max_iter=500,
        )
        self._scaler        = StandardScaler()
        self._cluster_labels: dict[int, str] = {}  # cluster_id → segment_label
        self._fitted        = False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only the clustering features that exist in df."""
        available = [c for c in _CLUSTER_FEATURES if c in df.columns]
        missing   = set(_CLUSTER_FEATURES) - set(available)
        if missing:
            logger.warning("Clustering features missing from DataFrame: %s", missing)
        return df[available].fillna(0)

    def _resolve_label_conflicts(self, raw_labels: dict[int, str]) -> dict[int, str]:
        """
        Ensure each segment label is assigned to at most one cluster.
        Ties are broken by which cluster has the highest centroid churn_probability.
        """
        # group clusters by assigned label
        label_to_clusters: dict[str, list[int]] = {}
        for cid, label in raw_labels.items():
            label_to_clusters.setdefault(label, []).append(cid)

        resolved: dict[int, str] = {}
        used_labels: set[str] = set()
        fallback_pool = list(SEGMENT_LABELS)

        for label, cids in label_to_clusters.items():
            if len(cids) == 1:
                resolved[cids[0]] = label
                used_labels.add(label)
            else:
                # Multiple clusters share a label — assign label to the one
                # with highest mean churn_probability centroid value
                centroids = self._kmeans.cluster_centers_
                feat_idx = _CLUSTER_FEATURES.index("churn_probability") if "churn_probability" in _CLUSTER_FEATURES else 0
                best = max(cids, key=lambda c: centroids[c][feat_idx])
                resolved[best] = label
                used_labels.add(label)
                # Give the others a fallback label
                for cid in cids:
                    if cid not in resolved:
                        # pick a label not yet used
                        for fl in fallback_pool:
                            if fl not in used_labels:
                                resolved[cid] = fl
                                used_labels.add(fl)
                                break
                        else:
                            resolved[cid] = "Loyal"  # final fallback

        # Assign any unresolved cluster ids
        for cid in range(self.n_clusters):
            if cid not in resolved:
                for fl in fallback_pool:
                    if fl not in used_labels:
                        resolved[cid] = fl
                        used_labels.add(fl)
                        break
                else:
                    resolved[cid] = "Loyal"

        return resolved

    def _label_clusters(self, feat_matrix: np.ndarray, feature_cols: list[str]) -> None:
        """Assign human labels to each KMeans cluster centroid."""
        raw: dict[int, str] = {}
        for cid, center in enumerate(self._kmeans.cluster_centers_):
            # Inverse-transform to original scale for interpretable rubric
            original_center = self._scaler.inverse_transform(center.reshape(1, -1))[0]
            centroid_dict = dict(zip(feature_cols, original_center))
            raw[cid] = _score_centroid(centroid_dict)
            logger.debug("Cluster %d centroid: %s → %s", cid, centroid_dict, raw[cid])

        self._cluster_labels = self._resolve_label_conflicts(raw)
        logger.info("Cluster labels: %s", self._cluster_labels)

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "CustomerSegmenter":
        """
        Fit the KMeans model on *df*.

        Parameters
        ----------
        df : DataFrame that includes the clustering features (post-prediction).
        """
        feat_df = self._select_features(df)
        feat_cols = list(feat_df.columns)

        X = self._scaler.fit_transform(feat_df.values)
        self._kmeans.fit(X)
        self._label_clusters(X, feat_cols)
        self._feature_cols = feat_cols
        self._fitted = True

        logger.info(
            "CustomerSegmenter fitted. %d clusters from %d rows.",
            self.n_clusters, len(df),
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign segment labels to each row in *df*.

        Returns a copy of *df* with two new columns:
        - ``cluster_id``   : int (raw KMeans cluster index)
        - ``segment``      : str (human-readable label)
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict().")

        feat_df = self._select_features(df)
        X = self._scaler.transform(feat_df.values)

        cluster_ids = self._kmeans.predict(X)
        segments    = [self._cluster_labels[cid] for cid in cluster_ids]

        out = df.copy()
        out["cluster_id"] = cluster_ids
        out["segment"]    = segments
        return out

    def segment_profile(self, df_segmented: pd.DataFrame) -> dict[str, dict]:
        """
        Compute per-segment descriptive statistics.

        Parameters
        ----------
        df_segmented : DataFrame already containing the ``segment`` column.

        Returns
        -------
        dict mapping segment_label → {metric: value}
        """
        if "segment" not in df_segmented.columns:
            raise ValueError("DataFrame must contain 'segment' column. Call .predict() first.")

        profile: dict[str, dict] = {}
        numeric_cols = [
            c for c in ["tenure", "MonthlyCharges", "InactiveDays",
                        "churn_probability", "risk_score", "service_count"]
            if c in df_segmented.columns
        ]

        for seg, grp in df_segmented.groupby("segment"):
            stats: dict[str, Any] = {
                "count":               int(len(grp)),
                "pct_of_total":        round(len(grp) / len(df_segmented) * 100, 1),
            }
            for col in numeric_cols:
                stats[f"avg_{col}"]    = round(float(grp[col].mean()), 2)
                stats[f"median_{col}"] = round(float(grp[col].median()), 2)

            if "churn_label" in grp.columns:
                stats["churn_count"] = int((grp["churn_label"] == "Churn").sum())
                stats["churn_rate"]  = round(
                    stats["churn_count"] / max(len(grp), 1) * 100, 1
                )

            profile[str(seg)] = stats

        return profile

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> Path:
        """Persist the fitted segmenter to disk."""
        dest = path or MODEL_DIR / _SEGMENTER_FILENAME
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, dest)
        logger.info("CustomerSegmenter saved → %s", dest)
        return dest

    @classmethod
    def load(cls, path: Path | None = None) -> "CustomerSegmenter":
        """Load a previously saved CustomerSegmenter from disk."""
        src = path or MODEL_DIR / _SEGMENTER_FILENAME
        if not src.exists():
            raise FileNotFoundError(
                f"Segmenter not found at {src}. Run the training pipeline first."
            )
        obj = joblib.load(src)
        logger.info("CustomerSegmenter loaded from %s", src)
        return obj


# ── Convenience wrapper ───────────────────────────────────────────────────────

def segment_customers(df: pd.DataFrame, save: bool = True) -> tuple[pd.DataFrame, CustomerSegmenter]:
    """
    Fit a new CustomerSegmenter on *df* and return (segmented_df, segmenter).

    Parameters
    ----------
    df   : DataFrame containing predictions (churn_probability must be present).
    save : If True, persist the fitted segmenter to MODEL_DIR.
    """
    segmenter = CustomerSegmenter()
    segmenter.fit(df)
    df_out = segmenter.predict(df)
    if save:
        segmenter.save()
    return df_out, segmenter
