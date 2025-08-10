import pandas as pd
from pathlib import Path
import rfmbinner
import random

class DataLoader:
    
    def __init__(self,path,customer_id,transaction_id,transaction_date,amount):
        """
        Initialize the DataLoader.

        Parameters:
            path (str): Path to the CSV file.
            customer_id (str): Column name for customer IDs.
            transactionid (str): Column name for transaction IDs.
            transaction_date (str): Column name for transaction dates.
            amount (str): Column name for transaction amounts.

        """
        self.path = path
        self.customer_id = customer_id
        self.transaction_id = transaction_id
        self.transaction_date = transaction_date 
        self.amount = amount 
    
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch raw CSV data from the specified path.

        Parameters:
            path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded data.
        """
        data_path = Path(self.path)
        if not data_path.exists():
            raise FileNotFoundError(f"CSV file not found at: {data_path}")
        data = pd.read_csv(data_path)
        data['CustomerID'] = [random.choice(list(range(1,12801))) for _ in range(len(data))]
        data.rename(columns={self.customer_id:'CustomerID',
                             self.transaction_id: 'TransactionID',
                             self.transaction_date:'Date',
                             self.amount: 'Amount'}, inplace=True)
        return data
    
    def calculate_rfm(self, snapshot_date: str, window: pd.Timedelta,df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RFM metrics for customer segmentation.

        Parameters:
            snapshot_date (str): Date to calculate recency from (YYYY-MM-DD).

        Returns:
            pd.DataFrame: RFM metrics per CustomerID.
        """
        data = df.copy()
        snapshot = pd.to_datetime(snapshot_date)
        data['Date'] = pd.to_datetime(data.Date)
        data  = data[(data.Date<snapshot) & (data.Date>(snapshot-window))]
        data['recency'] = (snapshot - pd.to_datetime(data['Date'])).dt.days
        print(data)
        rfm = data.groupby('CustomerID').agg({
            'recency': 'min',
            'TransactionID': 'count',               # frequency: number of transactions
            'Amount': 'sum'             # monetary: total spend
        }).reset_index()

        rfm.rename(columns={'TransactionID': 'frequency', 'Amount': 'monetary'}, inplace=True)
        rfm['date'] = snapshot_date

        return rfm[['CustomerID', 'date', 'recency', 'frequency', 'monetary']]
    
    def calculate_target(self,date: str, window_size: pd.Timedelta, \
                         repurchase_threshold: float, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculating the target for the model
        Parameters:
            date (str): Date to calculate recency from (YYYY-MM-DD).
            window_size (pd.Timedelta): Number of days in which to consider repurchase
            repurchase_threshold: Minimum spend to be considered a repurchase
        Returns:
            pd.DataFrame: Target per CustomerID.
        """
        #caculate whether a customer made total purchases after date exceeding repurchase threshold
        data = df.copy()
        data['Date'] = pd.to_datetime(data['Date'])
        date = pd.to_datetime(date)
        future_purchases = data[(data['Date'] > date) & \
                                (data['Date'] <= date + window_size)]

        target_customers = future_purchases.groupby('CustomerID')[['Amount']].sum()
        target_customers = target_customers[target_customers >= repurchase_threshold].index

        total_future_purchases = future_purchases.groupby('CustomerID')[['Amount']].sum()
        data = data.merge(total_future_purchases.rename(columns={'Amount': 'subsequent_purchases'}),
                            on='CustomerID', how='left')
        data['subsequent_purchases'] = data['subsequent_purchases'].fillna(0)


        data['target'] = data['CustomerID'].isin(target_customers).astype(int)
        data['date'] = date
        return data[['CustomerID','date','target','subsequent_purchases']].drop_duplicates()
    
    def rfm_segments(self,date: str, window: pd.Timedelta, df: pd.DataFrame) -> pd.DataFrame:
        """
        Segments customers based on their RFM score.

        Parameters:
            date (str): Date to calculate recency from (YYYY-MM-DD).
            df (pd.DataFrame): Raw transactional data.

        Returns:
            pd.DataFrame: RFM segments per CustomerID.
        """
        data = df.copy()
        data = self.calculate_rfm(date,window,data)
        # Create RFM scores
        data['r_score'] = pd.qcut(data['recency'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1])
        data['f_score'] = pd.qcut(data['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        data['m_score'] = pd.qcut(data['monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        data['rfm_score'] = data['r_score'].astype(str) + \
                            data['f_score'].astype(str) + \
                            data['m_score'].astype(str)
        data['rfm_score_int'] = data['r_score'].astype(int) + \
                            data['f_score'].astype(int)+ \
                            data['m_score'].astype(int)           
        #create a dicitionary with the rfm_score that maps into segment, the key must be a regex
        segment_map = {
                    r'^55[1-5]$': 'Champions',
                    r'^54[1-5]$': 'Loyal Customers',
                    r'^45[1-5]$': 'Potential Loyalists',
                    r'^53[1-5]$': 'New or Returning Customers',
                    r'^33[1-5]$': 'Promising',
                    r'^22[1-5]$': 'Needs Attention',
                    r'^\d{3}$': 'Others'  # Matches any other 3-digit score
                }
        data['segment'] = data['rfm_score'].replace(segment_map, regex=True)
        data['date'] = date
        return data[['CustomerID', 'date', 'rfm_score','rfm_score_int', 'segment']]
    
    
if __name__=='__main__':
    loader = DataLoader('data\Amazon Sale Report.csv','CustomerID','Order ID','Date','Amount')
    orig_data = loader.fetch_data()
    rfm = loader.calculate_rfm('2022-07-31',pd.Timedelta(days=99999),orig_data)
    print(rfm.head())
