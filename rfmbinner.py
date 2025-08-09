import pandas as pd
import numpy as np

class RFMBinner:
    def __init__(self, n_bins=5):
        """
        Initializes the RFMBinner with a specified number of bins.

        Parameters:
            n_bins (int): The number of bins to use for RFM segmentation.
        """
        self.n_bins = n_bins
        self.dictionary = {} # dictionary is here to save the cuts for each measure of RFM 

    def fit(self, df: pd.DataFrame):
        self.dictionary = {}
        for col in ['Recency', 'Frequency', 'Monetary']:
            series = pd.to_numeric(df[col], errors='coerce').dropna()

            try:
                _, bins = pd.qcut(series, q=self.n_bins, retbins=True, duplicates='drop')
                bins = np.unique(np.round(bins, 8))
                bins[0] = series.min()
                bins[-1] = series.max()
                self.dictionary[col] = bins.tolist()
            except ValueError:
                self.dictionary[col] = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.dictionary:
            raise ValueError("Must call fit() before transform().")

        result = df.copy()
        for col in ['Recency', 'Frequency', 'Monetary']:
            edges = self.dictionary[col]
            if edges is not None:
                result[f'{col}_Bin'] = pd.cut(df[col], bins=edges, labels=False, include_lowest=True) + 1
                if col == 'Recency':
                    result[f'{col}_Bin'] = self.n_bins + 1 - result[f'{col}_Bin']
            else:
                result[f'{col}_Bin'] = None
        return result

if __name__=='__main__':
    # Example Usage:
    data = {
        'CustomerID': [1, 2, 3, 4, 5],
        'Recency': [10, 150, 30, 200, 50],
        'Frequency': [5, 1, 8, 2, 10],
        'Monetary': [1000, 0, 1500, 300, 2000]
    }
    df = pd.DataFrame(data)

    binner = RFMBinner(n_bins=5)
    binner.fit(df)
    df_binned = binner.transform(df)
    print(df_binned)

    # Test with a different number of bins
    binner_3_bins = RFMBinner(n_bins=3)
    binner_3_bins.fit(df)
    df_binned_3 = binner_3_bins.transform(df)
    print(df_binned_3)

    # Test with data that has fewer unique values than bins
    data_sparse = {
        'CustomerID': [1, 2, 3],
        'Recency': [10, 20, 10],
        'Frequency': [1, 1, 2],
        'Monetary': [100, 200, 100]
    }
    df_sparse = pd.DataFrame(data_sparse)
    binner_sparse = RFMBinner(n_bins=5)
    binner_sparse.fit(df_sparse)
    df_binned_sparse = binner_sparse.transform(df_sparse)
    print(df_binned_sparse)