
import numpy as np
import pandas as pd
from dataloader import DataLoader
import xgboost as xgb
import dataloader
import joblib

class VintageCreator:

    def __init__(self,date: str):
        self.date = date
    
    def score_data(self):

        loader = dataloader.DataLoader(path=r'data\customer_transaction_data.csv',customer_id='CustomerID',transaction_id='TransactionID',transaction_date='PurchaseDate',amount='TotalAmount')
        self.trans_data = loader.fetch_data()
        self.trans_data['Date'] = pd.to_datetime(self.trans_data['Date'])
        self.date = pd.to_datetime(self.date)
        rfm_data = loader.calculate_rfm(self.date,pd.Timedelta(days=99999),self.trans_data)
        segments = loader.rfm_segments(self.date,pd.Timedelta(days=99999),self.trans_data)
        rfm_data = rfm_data.merge(segments,on=['CustomerID','date'],how='left')
        dem_data = loader.dedup_demographic_variables(self.trans_data)
        model_data = rfm_data.merge(dem_data,on='CustomerID',how='left')
        columns_for_model_data = ['CustomerID','date','recency',\
                                  'frequency','monetary',\
                                    'Gender','Age','Province',\
                                        'rfm_score_int']
        model = xgb.Booster()
        model.load_model(r'data\PropensityToBuy.json')
        model_data = model_data[columns_for_model_data].set_index(['CustomerID','date'])
        encoder = joblib.load(r'data\encoder.pkl')
        model_data = encoder.transform(model_data)
        model_data['Score'] = model.predict(xgb.DMatrix(model_data))
        bins = joblib.load(r'data\score_bins.pkl')
        model_data['Score_Segments'] = pd.cut(model_data['Score'], bins=bins)
        return model_data

    #create a method that calculates the vintage curve per Score Segment based on amount from 
    #transactional data 
    def create_vintage(self) -> pd.DataFrame:
        scored_data = self.score_data()

        # Filter transactions to include only those after the snapshot date
        future_transactions = self.trans_data[self.trans_data['Date'] > self.date].copy()

        # Merge scored data with future transactions
        merged_data = future_transactions.merge(
            scored_data[['Score_Segments']],
            on='CustomerID',
            how='inner'
        )

        # Calculate weeks since snapshot date
        merged_data['Weeks_Since_Snapshot'] = ((merged_data['Date'] - self.date).dt.days // 7).astype(int)

        # Aggregate weekly spend per segment
        weekly_spend = merged_data.groupby(
            ['Score_Segments', 'Weeks_Since_Snapshot']
        )['Amount'].sum().reset_index()

        # Pivot to wide format
        pivot = weekly_spend.pivot(
            index='Weeks_Since_Snapshot',
            columns='Score_Segments',
            values='Amount'
        ).fillna(0)

        # Sort index and apply cumulative sum to ensure monotonicity
        pivot = pivot.sort_index().cumsum()

        # Ensure all weeks are present
        max_week = pivot.index.max()
        all_weeks = pd.Series(range(max_week + 1), name='Weeks_Since_Snapshot')
        pivot = pivot.reindex(all_weeks, fill_value=0)

        return pivot


if __name__=='__main__':
    vintage = VintageCreator('2024-08-30')
    vintage_curve = vintage.create_vintage()
    print(vintage_curve)
    













































































































































































































































