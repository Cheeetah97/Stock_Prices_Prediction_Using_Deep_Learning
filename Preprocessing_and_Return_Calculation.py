#%%
# Importing the Libraries
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

def calculate_return(series):
    return (series.shift(-1)/series)-1

def calculate_nfuture_sum(sum_horizon,series):
    p_sum = []
    for n in range(1,len(series)+1,1):
        curr = series.iloc[[n-1]]
        next = series.iloc[n:n+sum_horizon]
        if len(next)==sum_horizon:
            p_sum.append(next.sum())
        else:
            p_sum.append(np.nan)
    return pd.Series(p_sum)
#%%
# Outlier Removal and Price Return Calculation

stock_data = pd.read_csv(data_path+"Stocks_Data_Raw.csv")
stock_data["Datetime"] = pd.to_datetime(stock_data["Datetime"],format="%Y-%m-%d %H:00:00")
stock_data["Hour"] = stock_data["Datetime"].dt.hour

stocks = []
low_counts = []
high_counts = []
df = pd.DataFrame()
for stk in list(stock_data.Stock.unique()):
    temp = stock_data[stock_data.Stock==stk]
    temp = temp.sort_values(by=["Datetime"])
    temp = temp.reset_index(drop=True)
    
    # Filling Missing Hours
    temp["Day"] = temp["Datetime"].dt.dayofyear
    for k,v in temp.iterrows():
        if (k > 0)&(temp["Day"].iloc[k]!=temp["Day"].iloc[k-1])&(temp["Hour"].iloc[k]!=8):
            index_pos = k
            new_rows = pd.concat([temp.iloc[[k]] for i in range(temp["Hour"].iloc[k],temp["Hour"].min(),-1)],ignore_index=True)
            for k,v in new_rows.iterrows():
                new_rows["Datetime"].iloc[k] = new_rows["Datetime"].iloc[k]-pd.DateOffset(hours=k+1)
                new_rows["Hour"].iloc[k] = new_rows["Hour"].iloc[k]-(k+1)
                new_rows.loc[k,["Open","High","Low","Close"]] = np.nan
    temp1 = temp.iloc[:index_pos]
    temp2 = temp.iloc[index_pos:]
    temp = pd.concat([temp1,new_rows.sort_values(by=["Datetime"]),temp2],ignore_index=True)
    temp["Open"] = temp["Open"].interpolate(method='polynomial',order=2)
    temp["High"] = temp["High"].interpolate(method='polynomial',order=2)
    temp["Low"] = temp["Low"].interpolate(method='polynomial',order=2)
    temp["Close"] = temp["Close"].interpolate(method='polynomial',order=2)
    temp = temp.drop("Day",axis=1)
    
    temp["Day_Diff"] = (temp["Datetime"].shift(9) - temp["Datetime"]).dt.days
    temp["Day_Diff"] = temp["Day_Diff"].fillna(-1)
    
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
        
        # Removing the Effect of Missing Days(Weekends)
        for i in range(len(temp)-1,2,-1):
            if (temp["Day_Diff"].iloc[i]<-1)&(temp["Hour"].iloc[i] == 8):
                prev = temp[["Hour","Day_Diff",col]].iloc[:i]
                thr = 0.020*np.mean(prev.loc[~((prev.Hour.isin([8,9]))&(prev.Day_Diff!=-1)),col])
                dif = temp[col].iloc[i]-temp[col].iloc[i-1]
                if (dif >= thr) or (dif <= -thr):
                    if dif > 0:
                        temp[col].iloc[:i] += dif - (thr/2)
                    else:
                        temp[col].iloc[:i] += dif + (thr/2)

        # Removing the Effect of Missing Hours of a Day
        temp[col+"_Sum"] = calculate_nfuture_sum(8,temp[col])
    
        # Calculating Price Returns
        temp[col[0]+"_S"] = calculate_return(temp[col+"_Sum"])
        temp[col[0]] = calculate_return(temp[col])
    
    df = pd.concat([df,temp])

corrections_df = pd.DataFrame(data={"Stock":stocks,"Low_Counts":low_counts,"High_Counts":high_counts})

df[["Datetime","Open","High","Low","Close","Stock",
    "Open_Sum","High_Sum","Low_Sum","Close_Sum"]].to_csv(data_path+"Stocks_Data(Prices).csv",index=False)

stock_data = df[["Datetime","Stock","O","H","L","C","O_S","H_S","L_S","C_S"]].reset_index(drop=True).copy()
stock_data = stock_data[stock_data.groupby(by=["Stock"]).cumcount(ascending=False) > 0]
stock_data.to_csv(data_path+"Stocks_Data.csv",index=False)
#%%
# Pivoting the Data and Saving
stock_data = pd.pivot_table(stock_data,values=["O","H","L","C","O_S","H_S","L_S","C_S"],index="Datetime",columns="Stock")
stock_data = stock_data.reset_index()
stock_data.columns = ['_'.join(col).strip() if col[0] != "Datetime" else ' '.join(col).strip() for col in stock_data.columns.values]
stock_data.to_csv(data_path+"Stocks_Data_Pivoted.csv",index=False)