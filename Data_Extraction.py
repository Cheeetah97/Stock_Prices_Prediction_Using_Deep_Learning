import pandas as pd
import yfinance as yf
from datetime import datetime

data_path = "M:/Dissertation/Data/"

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
# Extracting Stock Data and converting Timezones and Currencies where necessary

stocks = {"Shell_PLC":"SHEL.L",
          "Total_Energies_SE":"TTE.L",
          "Bp_PLC":"BP.L",
          "Glencore_PLC":"GLEN.L",
          "Harbour_Energy":"HBR.L",
          "Energean_PLC":"ENOG.L",
          "Thungela_Resources":"TGA.L",
          "ITM_Power":"ITM.L",
          "Ceres_Power":"CWR.L",
          "Tullow_Oil_PLC":"TLW.L",
          "Diversified_Energy":"DEC.L",
          "Serica_Energy":"SQZ.L"}

currencies = {"Shell_PLC":"GBp",
              "Total_Energies_SE":"EUR",
              "Bp_PLC":"GBp",
              "Glencore_PLC":"GBp",
              "Harbour_Energy":"GBp",
              "Energean_PLC":"GBp",
              "Thungela_Resources":"GBp",
              "ITM_Power":"GBp",
              "Ceres_Power":"GBp",
              "Tullow_Oil_PLC":"GBp",
              "Diversified_Energy":"GBp",
              "Serica_Energy":"GBp"}


start_date = "2022-01-31"
end_date = datetime.today() - pd.DateOffset(days=1)
time_format = "%Y-%m-%d %H:%M:%S%z"

stocks_df = pd.DataFrame()
for stk in stocks.items():
    stock_data = extract_data(stk[1],start_date,end_date).reset_index()[["Datetime","Open","High","Low","Close"]]
    stock_data["Datetime"] = stock_data["Datetime"].apply(lambda x:convert_to_utc(x,time_format))
    if currencies[stk[0]] != "GBp":
        currency_data = extract_data(str(currencies[stk[0]])+"GBP=X",start_date,end_date)
        currency_data = currency_data.reset_index()[["Datetime","Open","High","Low","Close"]]
        currency_data = currency_data.rename(columns={"Open":"Open"+str(currencies[stk[0]])+"GBP=X",
                                                      "High":"High"+str(currencies[stk[0]])+"GBP=X",
                                                      "Low":"Low"+str(currencies[stk[0]])+"GBP=X",
                                                      "Close":"Close"+str(currencies[stk[0]])+"GBP=X"})
        currency_data["Datetime"] = currency_data["Datetime"].apply(lambda x:convert_to_utc(x,time_format))
        stock_data = stock_data.merge(currency_data,how="left",on=["Datetime"])
        stock_data["Open"] = stock_data["Open"] * stock_data["Open"+str(currencies[stk[0]])+"GBP=X"] * 100
        stock_data["High"] = stock_data["High"] * stock_data["High"+str(currencies[stk[0]])+"GBP=X"] * 100
        stock_data["Low"] = stock_data["Low"] * stock_data["Low"+str(currencies[stk[0]])+"GBP=X"] * 100
        stock_data["Close"] = stock_data["Close"] * stock_data["Close"+str(currencies[stk[0]])+"GBP=X"] * 100
        stock_data = stock_data[["Datetime","Open","High","Low","Close"]]
    stock_data["Stock"] = stk[1]
    stocks_df = pd.concat([stocks_df,stock_data])
    
print(stocks_df["Stock"].value_counts())
#stocks_df.to_csv(data_path+"Stocks_Data_UnPivoted.csv",index=False)
        



