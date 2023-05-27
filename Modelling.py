#%%
import pandas as pd
import numpy as np
import tensorflow as tf

data_path = "M:/Dissertation/Data/"
#%%

stock_data = pd.read_csv(data_path+"Stocks_Data.csv")
stock_data["Datetime"] = pd.to_datetime(stock_data["Datetime"],format="%Y-%m-%d %H:00:00")

stock_data = stock_data.loc[stock_data.Stock=="BP.L"].reset_index(drop=True)
stock_data = stock_data.sort_values(by=["Datetime"])
stock_data = stock_data.drop("Stock",axis=1)

# pos = stock_data.loc[stock_data.C>0,"C"]
# neg = stock_data.loc[stock_data.C<0,"C"]
# %%
