import numpy as np
import pandas as pd

class OneRuleClassifier:
    """Simple implementation of 1‑Rule (OneR) classifier.
    Numeric features are discretised into N quantile buckets.
    """
    def __init__(self, n_bins=5, default_class=0):
        self.n_bins = n_bins
        self.default_class = default_class
        self.best_feature_ = None
        self.rules_ = None

    def _prepare_feature(self, series: pd.Series):
        if series.dtype.kind in 'bifc':
            # numeric -> quantile bins
            try:
                binned = pd.qcut(series, q=self.n_bins, duplicates='drop')
            except ValueError:
                # not enough unique values
                binned = series
            return binned.astype(str)
        else:
            return series.astype(str)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        best_err = np.inf
        best_feat = None
        best_rules = None

        for col in X.columns:
            feat = self._prepare_feature(X[col])
            df = pd.DataFrame({'feat': feat, 'target': y})
            rules = {}
            err = 0
            for val, group in df.groupby('feat'):
                maj_class = group['target'].value_counts().idxmax()
                rules[val] = maj_class
                err += (group['target'] != maj_class).sum()
            if err < best_err:
                best_err = err
                best_feat = col
                best_rules = rules

        self.best_feature_ = best_feat
        self.rules_ = best_rules
        return self

    def predict(self, X: pd.DataFrame):
        if self.best_feature_ is None:
            raise RuntimeError("Model not fitted")
        feat_col = X[self.best_feature_]
        # Use same discretization as training
        prepared_feat = self._prepare_feature(feat_col)
        preds = prepared_feat.map(self.rules_).fillna(self.default_class)
        return preds.astype(int).values