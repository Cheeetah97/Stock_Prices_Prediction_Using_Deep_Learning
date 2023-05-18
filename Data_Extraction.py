import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import pytz

#%%
# Functions 

# Data Extraction Function
def extract_data(ticker_symbol,start_date,end_date):
    data = yf.download(ticker_symbol,start=start_date,end=end_date,interval="1h")
    return data

# Time Zone Conversion Function
def convert_to_utc(local_time,time_format):
    time_zone_offset = local_time.strftime('%z')
    if time_zone_offset[0] == "+":
        utc_time = datetime.strptime(local_time.strftime("%Y-%m-%d %H:%M:%S"),"%Y-%m-%d %H:%M:%S") - pd.DateOffset(hours=int(time_zone_offset)//100)
    else:
        utc_time = datetime.strptime(local_time.strftime("%Y-%m-%d %H:%M:%S"),"%Y-%m-%d %H:%M:%S") + pd.DateOffset(hours=int(time_zone_offset)//100)
    return utc_time

#%%
# List all the Stocks

stocks = {"Shell_PLC":"SHEL.L",
          "Total_Energies_SE":"TTE.L",
          "Bp_PLC":"BP.L",
          "Woodside_Energy":"WDS.L",
          "Harbour_Energy":"HBR.L",
          "Energean_PLC":"ENOG.L",
          "Thungela_Resources":"TGA.L",
          "ITM_Power":"ITM.L",
          "Ceres_Power":"CWR.L"}

currencies = {"Shell_PLC":"GBp",
              "Total_Energies_SE":"EUR",
              "Bp_PLC":"GBp",
              "Woodside_Energy":"GBp",
              "Harbour_Energy":"GBp",
              "Energean_PLC":"GBp",
              "Thungela_Resources":"GBp",
              "ITM_Power":"GBp",
              "Ceres_Power":"GBp"}


start_date = "2021-07-31"#datetime.today() - pd.DateOffset(days=729)
end_date = datetime.today()
time_format = "%Y-%m-%d %H:%M:%S%z"

stocks_df = pd.DataFrame()
for stk in stocks.items():
    stock_data = extract_data(stk[1],start_date,end_date).reset_index()[["Datetime","Open"]]
    stock_data["Datetime"] = stock_data["Datetime"].apply(lambda x:convert_to_utc(x,time_format))
    if currencies[stk[0]] != "GBp":
        currency_data = extract_data(str(currencies[stk[0]])+"GBP=X",start_date,end_date).reset_index()[["Datetime","Open"]].rename(columns={"Open":str(currencies[stk[0]])+"GBP=X"})
        currency_data["Datetime"] = currency_data["Datetime"].apply(lambda x:convert_to_utc(x,time_format))
        stock_data = stock_data.merge(currency_data,how="left",on=["Datetime"])
        stock_data["Open"] = stock_data["Open"] * stock_data[str(currencies[stk[0]])+"GBP=X"] * 100
        stock_data = stock_data[["Datetime","Open"]]
    stock_data["Stock"] = stk[1]
    stocks_df = pd.concat([stocks_df,stock_data])
    
print(stocks_df["Stock"].value_counts())
        
        



