
import numpy as np
import pandas as pd
from dataloader import DataLoader
import xgboost as xgb
import dataloader
import joblib

class VintageCreator:

    def __init__(self,date: str):
        self.date = date
    
    def create_vintage(self):

        loader = dataloader.DataLoader(path=r'data\customer_transaction_data.csv',customer_id='CustomerID',transaction_id='TransactionID',transaction_date='PurchaseDate',amount='TotalAmount')
        self.trans_data = loader.fetch_data()
        self.trans_data['Date'] = pd.to_datetime(self.trans_data['Date'])
        self.data = pd.to_datetime(self.date)
        rfm_data = loader.calculate_rfm(self.date,pd.Timedelta(days=99999),self.trans_data)
        segments = loader.rfm_segments(self.date,pd.Timedelta(days=99999),self.trans_data)
        rfm_data = rfm_data.merge(segments,on=['CustomerID','date'],how='left')
        dem_data = loader.dedup_demographic_variables(self.trans_data)
        model_data = rfm_data.merge(dem_data,on='CustomerID',how='left')
        columns_for_model_data = ['CustomerID','date','recency',\
                                  'frequency','monetary',\
                                    'Gender','Age','Province',\
                                        'rfm_score_int']
        model = xgb.XGBClassifier()
        model.load_model(r'd:\\VS CODE PROJECTS\\CLVM\\data\\PropensityToBuy.json')
        model_data = model_data[columns_for_model_data].set_index(['CustomerID','date'])
        encoder = joblib.load(r'data\encoder.pkl')
        model_data = encoder.transform(model_data)
        y_probs = model.predict(model_data)
        bins = joblib.load(r'data\score_bins.pkl')
        model_data['Score_Segments'] = pd.cut(y_probs, bins=bins, labels=False,include_lowest=True)

        data = self.trans_data.copy()
        first_purchase_date = data.groupby('CustomerID')['Date'].min().reset_index()
        data = data.merge(first_purchase_date.rename(columns={'Date': 'FirstPurchaseDate'}), on='CustomerID', how='left')
        data['Vintage'] = ((data['Date'] - data['FirstPurchaseDate'])/7).round()
        data.drop(columns=['FirstPurchaseDate'], inplace=True)
        data = data.merge(model_data[['Score_Segments']], on='CustomerID', how='left')
        return data.groupby(['Score_Segments','Vintage'])['Amount'].cumsum()
    

if __name__=='__main__':
    vintage_creator = VintageCreator(date='2023-07-31')
    vintage_data = vintage_creator.create_vintage()
    print(vintage_data.head())