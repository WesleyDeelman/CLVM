import pandas as pd
import numpy as np
import random
from faker import Faker

# Setup
fake = Faker('en_GB')
np.random.seed(42)
random.seed(42)

# Parameters
num_customers = 10000
num_transactions = 100000
categories = ['Electronics', 'Clothing', 'Groceries', 'Home & Garden', 'Health & Beauty', 'Sports', 'Books']
provinces = ['Western Cape', 'Gauteng', 'KwaZulu-Natal', 'Eastern Cape', 'Free State', 'Limpopo', 'Mpumalanga', 'North West', 'Northern Cape']
genders = ['Male', 'Female']
payment_methods = ['Credit Card', 'Debit Card', 'Cash', 'EFT', 'Mobile Wallet']
purchase_channels = ['Online', 'In-Store', 'App']

# Generate customer pool
customer_pool = [{
    'CustomerID': f"CUST-{i:05d}",
    'Age': random.randint(18, 70),
    'Province': random.choice(provinces),
    'Gender': random.choice(genders)
} for i in range(num_customers)]

# Generate transactions
transactions = []
for _ in range(num_transactions):
    customer = random.choice(customer_pool)
    transaction = {
        'CustomerID': customer['CustomerID'],
        'TransactionID': f"TRANS-{_ + 1:07d}",
        'TransactionDate': fake.date_time_between(start_date='-3y', end_date='now').strftime('%Y-%m-%d %H:%M:%S'),
        'TransactionAmount': round(np.random.exponential(scale=500), 2),
        'ProductCategory': random.choice(categories),
        'Age': customer['Age'],
        'Province': customer['Province'],
        'Gender': customer['Gender'],
        'PaymentMethod': random.choices(payment_methods, weights=[0.3, 0.3, 0.2, 0.1, 0.1])[0],
        'PurchaseChannel': random.choices(purchase_channels, weights=[0.4, 0.5, 0.1])[0],
    }
    transaction['Store'] = fake.company() if transaction['PurchaseChannel'] == 'In-Store' else None
    transactions.append(transaction)

# Create DataFrame and save
df = pd.DataFrame(transactions)
reference_table = df.groupby('CustomerID')['TransactionID'].count().reset_index()
customerids = reference_table[reference_table.TransactionID>=2]['CustomerID']
df['TransactionAmount'] = df.apply(lambda x: x['TransactonAmount']*3 if x['CustomerID'] in customerids else x['TransactionAmount'], axis=1)
df.to_csv(r'data\synthetic_transactions.csv', index=False)
