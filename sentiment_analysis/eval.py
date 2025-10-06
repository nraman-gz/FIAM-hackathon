import pandas as pd
import numpy as np
import warnings, os
from datetime import datetime
warnings.simplefilter("ignore")

'''
this script evaluates a simple trading strategy based on sentiment delta scores
it simulates investing $10,000 in stocks with delta_score > 0.1 each month and calculates the overall return
'''

df = pd.read_csv("processed_sentiment.csv")
from datetime import datetime
from dateutil.relativedelta import relativedelta

stock_df = pd.read_csv("ret_sample.csv")

df = pd.read_csv("processed_sentiment.csv")
df['delta_score'].corr(df['ret_1_0'])
df["date"] = pd.to_datetime(df["date"])
df["date"] = df["date"].dt.strftime("%Y%m%d")


def monthly_strategy(month,stock_df):
    initial_investment = 10000
    cash_spent = 0
    total_cash = 0
    count = 0
    signals = df.sort_values('date')
    signals['date'] = signals['date'].astype(int)
    signals = signals[(signals['date'] > int(f"{month}01")) & (signals['date'] < int(f"{month+1}01"))]
    stock_df = stock_df[(stock_df['date'] > int(f"{month}01")) & (stock_df['date'] < int(f"{month+1}01"))]

    for row in signals.itertuples():
        initial_investment = 10000
        if row.delta_score > 0.1:
            initial_investment = initial_investment * (1 + row.ret_1_0)
            if str(initial_investment)=="nan":
                continue
            total_cash += initial_investment
            cash_spent += 10000
            count += 1
            #print(f"Bought {row.gvkey} at {row.prc} on {row.date}, new investment value: {initial_investment} ")
    #print(f"Month: {month}, total cash: {round(total_cash,2)}, cash spent: {cash_spent}, count: {count}")
    if cash_spent == 0:
        print(f"Month: {month}, total cash: {round(total_cash,2)}, cash spent: {cash_spent}, count: {count} return 0")
        return 0
    print(f"Month: {month}, total cash: {round(total_cash,2)}, cash spent: {cash_spent}, count: {count}, return {round((total_cash - cash_spent) / cash_spent * 100,2)}%")
    return (total_cash - cash_spent) / cash_spent

start_date = datetime(2015, 1, 1)
end_date = datetime(2025, 6, 1)
monthly_periods = pd.period_range(start=start_date, end=end_date, freq='M')
initial = 1_000_000
for i in list(monthly_periods):
    month = int(i.to_timestamp().to_pydatetime().strftime("%Y%m%d")[0:6])
    initial = initial * float (monthly_strategy(month, stock_df) + 1)
    #print(month,"return", monthly_strategy(month, stock_df))
    #break
print("Final cash after all months:", initial)
