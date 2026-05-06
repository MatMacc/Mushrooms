import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import importlib
import data_preprocessing_prova 

importlib.reload(data_preprocessing_prova)

from data_preprocessing_prova import DataProcessor

class DataProcessorTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        missing_strategy="mean",
        missing_columns=None,
        binary_cols=None,
        binary_positive=None,   # dict: {col: positive_value}
        onehot_cols=None,
        onehot_drop="first",
        scale_cols=None,
        scale_method="standard",
        variance_threshold=None,  # None = skip, else float
        log_cols=None,
        outlier_iqr_cols=None,
        outlier_std_cols=None,
        outlier_std_n=3
    ):
        self.missing_strategy = missing_strategy
        self.missing_columns = missing_columns

        self.binary_cols = binary_cols
        self.binary_positive = binary_positive or {}

        self.onehot_cols = onehot_cols
        self.onehot_drop = onehot_drop

        self.scale_cols = scale_cols
        self.scale_method = scale_method

        self.variance_threshold = variance_threshold
        self.log_cols = log_cols

        self.outlier_iqr_cols = outlier_iqr_cols
        self.outlier_std_cols = outlier_std_cols
        self.outlier_std_n = outlier_std_n

    def fit(self, X, y=None):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        

        self.processor_ = DataProcessor(X_df)

        # IMPORTANT: outlier removal changes row count -> not recommended inside sklearn Pipeline
        # because y would need to be filtered as well. So we skip it in pipeline fit/transform.
        # (Keep outlier removal outside the pipeline.)

        if self.missing_columns:
            self.processor_.fit_missing(strategy=self.missing_strategy, columns=self.missing_columns)
            self.processor_.transform_missing(columns=self.missing_columns)

        if self.log_cols:
            self.processor_.log_transform(self.log_cols)

        if self.binary_cols:
            for c in self.binary_cols:
                pos = self.binary_positive.get(c, None)
                self.processor_.map_binary(c, positive_value=pos)

        if self.onehot_cols:
            self.processor_.fit_onehot(self.onehot_cols, drop=self.onehot_drop)
            self.processor_.transform_onehot()

        if self.scale_cols:
            self.processor_.fit_scale(self.scale_cols, method=self.scale_method)
            self.processor_.transform_scale()

        if self.variance_threshold is not None:
            self.processor_.fit_variance_threshold(threshold=self.variance_threshold)
            self.processor_.transform_variance_threshold()

        # Save final feature names
        self.feature_names_out_ = list(self.processor_.get_df().columns)
        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        proc = DataProcessor(X_df)

        # reuse fitted objects from self.processor_
        if self.missing_columns:
            proc.imputers_ = self.processor_.imputers_
            proc.transform_missing(columns=self.missing_columns)

        if self.log_cols:
            proc.log_transform(self.log_cols)

        if self.binary_cols:
            for c in self.binary_cols:
                pos = self.binary_positive.get(c, None)
                proc.map_binary(c, positive_value=pos)

        if self.onehot_cols:
            proc.ohe_ = self.processor_.ohe_
            proc.ohe_cols_ = self.processor_.ohe_cols_
            proc.ohe_feature_names_ = self.processor_.ohe_feature_names_
            proc.transform_onehot()

        if self.scale_cols:
            proc.scaler_ = self.processor_.scaler_
            proc.scale_cols_ = self.processor_.scale_cols_
            proc.transform_scale()

        if self.variance_threshold is not None:
            proc.vt_ = self.processor_.vt_
            proc.vt_numeric_cols_ = self.processor_.vt_numeric_cols_
            proc.vt_selected_cols_ = self.processor_.vt_selected_cols_
            proc.transform_variance_threshold()

        # return numpy array for sklearn estimators
        return proc.get_df().to_numpy()

    def get_feature_names_out(self, input_features=None):
        return getattr(self, "feature_names_out_", None)