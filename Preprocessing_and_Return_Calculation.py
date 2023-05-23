import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_path = "M:/Dissertation/Data/"
#%%
# Functions

def calculate_bollinger_bands(series,window_size=9*7,num_std=5):
    rolling_mean = series.iloc[:-1].rolling(window=window_size).mean()
    rolling_std = series.iloc[:-1].rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band,lower_band,rolling_mean

def calculate_price_return(series):
    return (series.shift(-1)/series)-1
#%%
# Outlier Removal and Price Return Calculation

stock_data = pd.read_csv(data_path+"Stocks_Data_Raw.csv")
stock_data["Datetime"] = pd.to_datetime(stock_data["Datetime"],format="%Y-%m-%d %H:00:00")

stocks = []
low_counts = []
high_counts = []
df = pd.DataFrame()
for stk in list(stock_data.Stock.unique()):
    temp = stock_data[stock_data.Stock==stk]
    temp = temp.sort_values(by=["Datetime"])
    temp = temp.reset_index(drop=True)
    
    for col in ["Open","High","Low","Close"]:
        temp["Upper_Band"],temp["Lower_Band"],temp["Rolling_Mean"] = calculate_bollinger_bands(temp[col])
        temp.loc[temp.Upper_Band.isnull(),"Upper_Band"] = temp.loc[temp.Upper_Band.isnull(),col]
        temp.loc[temp.Lower_Band.isnull(),"Lower_Band"] = temp.loc[temp.Lower_Band.isnull(),col]
        
        stocks.append(stk+"_"+col)
        high_counts.append(len(temp.loc[temp[col]>temp.Upper_Band,col]))
        low_counts.append(len(temp.loc[temp[col]<temp.Lower_Band,col]))
        
        # Removing Outliers
        temp.loc[temp[col]>temp.Upper_Band,col] = temp["Rolling_Mean"]
        temp.loc[temp[col]<temp.Lower_Band,col] = temp["Rolling_Mean"]
    
        # Calculating Price Return
        temp[col[0]] = calculate_price_return(temp[col])
    
    temp = temp.drop(["Lower_Band","Upper_Band","Rolling_Mean"],axis=1)
    df = pd.concat([df,temp])

df[["Datetime","Open","High","Low","Close","Stock"]].to_csv(data_path+"Stocks_Data(Prices).csv",index=False)
df[["Datetime","O","H","L","C","Stock"]].to_csv(data_path+"Stocks_Data.csv",index=False)

stock_data = df[["Datetime","O","H","L","C","Stock"]].reset_index(drop=True).copy()
corrections_df = pd.DataFrame(data={"Stock":stocks,"Low_Counts":low_counts,"High_Counts":high_counts})
#%%
stock_data = pd.pivot_table(stock_data,values=["O","H","L","C"],index="Datetime",columns="Stock")
stock_data = stock_data.reset_index()
stock_data.columns = ['_'.join(col).strip() if col[0] != "Datetime" else ' '.join(col).strip() for col in stock_data.columns.values]
stock_data.to_csv(data_path+"Stocks_Data_Pivoted.csv",index=False)