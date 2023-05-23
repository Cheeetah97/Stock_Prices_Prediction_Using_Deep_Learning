import pandas as pd
import numpy as np

data_path = "M:/Dissertation/Data/"


# stock_data = pd.read_csv(data_path+"Stocks_Data.csv")
# stock_data = stock_data.loc[stock_data.Stock=="TGA.L"]
# stock_data["Datetime"] = pd.to_datetime(stock_data["Datetime"],format="%Y-%m-%d %H:00:00")


# pos = stock_data.loc[stock_data.C>0,"C"]
# neg = stock_data.loc[stock_data.C<0,"C"]

#%%
data = pd.read_csv(data_path+"Stocks_Data(Prices).csv")
data = data.loc[data.Stock=="BP.L"]
data = data[["Datetime","Open","Stock"]]
data = data.reset_index(drop=True)

data["Datetime"] = pd.to_datetime(data["Datetime"],format="%Y-%m-%d %H:00:00")
data["Day"] = data["Datetime"].dt.day
data["Hour"] = data["Datetime"].dt.hour
data["Month"] = data["Datetime"].dt.month

#data["Diff_1"] = data["Open"].shift(1) - data["Open"]
#data = data.dropna(subset=["Diff_1"])
#data["Diff_1"] = np.abs(data["Diff_1"])

for stk in list(data.Stock.unique()):
    temp = data[data.Stock==stk]
    temp = temp.sort_values(by=["Datetime"])
    temp = temp.reset_index(drop=True)
    
    for i in range(len(temp)-1,2,-1):
        if (temp["Hour"].iloc[i] == 8)
        thr = 0.10*np.mean(temp["Target"].iloc[:i])
        difa = temp["Target"].iloc[i]-temp["Target"].iloc[i-1]
        if (difa >= thr) or (difa <= -thr):  
            if difa > 0:
                temp["Target"].iloc[:i] += difa - 0.0045 
            else:
                temp["Target"].iloc[:i] += difa + 0.0043

#data2 = data.groupby(by=["Hour"],as_index=False)["Diff_1"].median()
