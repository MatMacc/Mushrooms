from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import pandas as pd

class DataProcessor:
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.scaler = None
        self.encoders = {}

    # =========================
    # CLEANING
    # =========================

    def remove_duplicates(self):
        self.df.drop_duplicates(inplace=True)
        return self

    def handle_missing(self, strategy='mean', columns=None):
        cols = columns if columns else self.df.columns

        for col in cols:
            if self.df[col].dtype in ['float64', 'int64']:
                imputer = SimpleImputer(strategy=strategy)
            else:
                imputer = SimpleImputer(strategy='most_frequent')

            self.df[col] = imputer.fit_transform(self.df[[col]]).ravel()

        return self

    def drop_missing(self, threshold=0.5):
        self.df.dropna(thresh=int((1 - threshold) * len(self.df)), inplace=True)
        return self

    def fix_categorical(self, column, mapping):
        self.df[column].replace(mapping, inplace=True)
        return self

    # =========================
    # OUTLIERS
    # =========================

    def remove_outliers_iqr(self, columns):
        if isinstance(columns, str):
            columns = [columns]

        mask = pd.Series(True, index=self.df.index)

        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            mask &= self.df[col].between(lower, upper)

        self.df = self.df.loc[mask].copy()
        return self

    def remove_outliers_std(self, columns, n_std=3):
        if isinstance(columns, str):
            columns = [columns]

        mask = pd.Series(True, index=self.df.index)

        for col in columns:
            mean = self.df[col].mean()
            std = self.df[col].std()

            lower = mean - n_std * std
            upper = mean + n_std * std

            mask &= self.df[col].between(lower, upper)

        self.df = self.df.loc[mask].copy()
        return self

    # =========================
    # FEATURE ENGINEERING
    # =========================

    def map_binary(self, column, positive_value=None):
        unique_vals = self.df[column].dropna().unique()

        if len(unique_vals) > 2:
            raise ValueError(f"{column} non è binaria")

        if positive_value is None:
            positive_value = unique_vals[0]

        self.df[column] = (self.df[column] == positive_value).astype(int)
        return self

    def log_transform(self, columns):
        self.df[columns] = np.log1p(self.df[columns])
        return self

    # =========================
    # ENCODING
    # =========================

    def encode_label(self, columns):
        for col in columns:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.encoders[col] = le

        return self

    def encode_onehot(self, columns):
        encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown="ignore")

        encoded = encoder.fit_transform(self.df[columns])
        new_cols = encoder.get_feature_names_out(columns)

        encoded_df = pd.DataFrame(encoded, columns=new_cols, index=self.df.index)

        self.df = pd.concat([self.df.drop(columns, axis=1), encoded_df], axis=1)

        self.encoders[tuple(columns)] = encoder
        return self

    # =========================
    # SCALING
    # =========================

    def scale(self, columns, method='standard'):
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Metodo non valido: usa 'standard' o 'minmax'")

        self.df.loc[:, columns] = self.scaler.fit_transform(self.df[columns])
        return self

    # =========================
    # FEATURE SELECTION
    # =========================

    def variance_threshold(self, threshold=0.0):
        numeric_df = self.df.select_dtypes(include=np.number)

        if numeric_df.shape[1] == 0:
            print("VarianceThreshold: nessuna colonna numerica trovata, salto.")
            return self

        selector = VarianceThreshold(threshold)

        # forza numpy float (più robusto)
        X_num = numeric_df.to_numpy(dtype=np.float64)
        selected = selector.fit_transform(X_num)

        selected_cols = numeric_df.columns[selector.get_support()]
        selected_df = pd.DataFrame(selected, columns=selected_cols, index=self.df.index)

        # tieni eventuali non numeriche (se ce ne sono)
        non_numeric = self.df.drop(columns=numeric_df.columns)

        self.df = pd.concat([selected_df, non_numeric], axis=1)
        return self

    # =========================
    # OUTPUT
    # =========================

    def get_df(self):
        return self.df.copy()

    # =========================
    # STATEFUL FIT/TRANSFORM HELPERS (for sklearn pipelines)
    # =========================
    def fit_missing(self, strategy='mean', columns=None):
        cols = columns if columns else self.df.columns
        self.imputers_ = getattr(self, "imputers_", {})

        for col in cols:
            if self.df[col].dtype in ['float64', 'int64', 'int32', 'float32']:
                imp = SimpleImputer(strategy=strategy)
            else:
                imp = SimpleImputer(strategy='most_frequent')

            imp.fit(self.df[[col]])
            self.imputers_[col] = imp
        return self

    def transform_missing(self, columns=None):
        cols = columns if columns else self.df.columns
        for col in cols:
            if col in self.imputers_:
                self.df[col] = self.imputers_[col].transform(self.df[[col]]).ravel()
        return self

    def fit_onehot(self, columns, drop='first'):
        self.ohe_cols_ = list(columns)
        self.ohe_ = OneHotEncoder(
            sparse_output=False,
            drop=drop,
            handle_unknown="ignore"
        )
        self.ohe_.fit(self.df[self.ohe_cols_])
        self.ohe_feature_names_ = list(self.ohe_.get_feature_names_out(self.ohe_cols_))
        return self

    def transform_onehot(self):
        encoded = self.ohe_.transform(self.df[self.ohe_cols_])
        encoded_df = pd.DataFrame(encoded, columns=self.ohe_feature_names_, index=self.df.index)
        self.df = pd.concat([self.df.drop(self.ohe_cols_, axis=1), encoded_df], axis=1)
        return self

    def fit_scale(self, columns, method='standard'):
        self.scale_cols_ = list(columns)
        if method == 'standard':
            self.scaler_ = StandardScaler()
        elif method == 'minmax':
            self.scaler_ = MinMaxScaler()
        else:
            raise ValueError("Metodo non valido: usa 'standard' o 'minmax'")

        self.scaler_.fit(self.df[self.scale_cols_])
        return self

    def transform_scale(self):
        self.df.loc[:, self.scale_cols_] = self.scaler_.transform(self.df[self.scale_cols_])
        return self

    def fit_variance_threshold(self, threshold=0.0):
        self.vt_threshold_ = threshold
        numeric_df = self.df.select_dtypes(include=np.number)

        self.vt_numeric_cols_ = list(numeric_df.columns)
        if len(self.vt_numeric_cols_) == 0:
            self.vt_ = None
            self.vt_support_ = None
            return self

        self.vt_ = VarianceThreshold(threshold)
        X_num = numeric_df.to_numpy(dtype=np.float64)
        self.vt_.fit(X_num)
        self.vt_support_ = self.vt_.get_support()
        self.vt_selected_cols_ = list(numeric_df.columns[self.vt_support_])
        return self

    def transform_variance_threshold(self):
        if getattr(self, "vt_", None) is None:
            return self

        numeric_df = self.df[self.vt_numeric_cols_]
        X_num = numeric_df.to_numpy(dtype=np.float64)
        selected = self.vt_.transform(X_num)

        selected_df = pd.DataFrame(selected, columns=self.vt_selected_cols_, index=self.df.index)
        non_numeric = self.df.drop(columns=self.vt_numeric_cols_)

        self.df = pd.concat([selected_df, non_numeric], axis=1)
        return self