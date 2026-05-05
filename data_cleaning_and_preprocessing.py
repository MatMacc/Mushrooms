import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold


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