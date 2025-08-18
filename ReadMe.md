## What is a Customer Lifetime Value (CLV) model?
""Customer Lifetime Value (CLV) is a metric that estimates the total revenue or profit a business can expect from a single customer throughout the entire duration of their relationship. It goes beyond individual transactions to capture the long-term financial impact of customer loyalty, repeat purchases, and even referrals. By understanding CLV, businesses can make smarter decisions about customer acquisition costs, retention strategies, and personalized marketing efforts—ultimately focusing on maximizing value from high-potential customers rather than short-term gains."" according to co-pilot.

Although the definition says that we calculate Value at he Client level, it is important to notice that ultimately we action this intelligence on high value clients. So ultimately it suffices to identify the high value, and circumventing the customer level calculation.

The entry point to the program is main.py.
Requirements.txt has all the necessary packages to run the app, I would suggest installing all the packges in a virtual environment.
Think of this code a template to building a read CLVM on real data. The script outputs 3 segment vintages ranked on their propensity to buy. There is also prediction curve fitted.