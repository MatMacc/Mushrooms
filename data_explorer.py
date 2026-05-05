import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class DataExplorer:
    def __init__(self, df: pd.DataFrame):
        # Copia del dataframe
        self.df = df.copy()

    def correlation_heatmap(self):
        # Mostra la matrice di correlazione tra variabili numeriche
        plt.figure(figsize=(10, 6))
        corr = self.df.corr(numeric_only=True)

        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show()

        return self

    def boxplot_outliers(self, columns=None):
        # Visualizza outliers tramite boxplot
        cols = columns if columns else self.df.select_dtypes(include=np.number).columns

        for col in cols:
            plt.figure()
            sns.boxplot(x=self.df[col])
            plt.title(f"Boxplot - {col}")
            plt.show()

        return self

    def distribution_plot(self, columns=None):
        # Visualizza distribuzione (utile per capire skewness e outliers)
        cols = columns if columns else self.df.select_dtypes(include=np.number).columns

        for col in cols:
            plt.figure()
            sns.histplot(self.df[col], kde=True)
            plt.title(f"Distribution - {col}")
            plt.show()

        return self

    def pairplot(self, columns=None):
        # Visualizza relazioni tra variabili (utile ma costoso)
        cols = columns if columns else self.df.select_dtypes(include=np.number).columns

        sns.pairplot(self.df[cols])
        plt.show()

        return self

    def missing_values_summary(self):
        # Mostra numero di valori mancanti per colonna
        missing = self.df.isnull().sum()
        print("Missing values per column:\n", missing)
        return self

    def get_df(self):
        # Restituisce il dataframe (non modificato)
        return self.df