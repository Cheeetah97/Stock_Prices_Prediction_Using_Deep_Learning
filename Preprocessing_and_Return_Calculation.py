import pandas as pd
import matplotlib.pyplot as plt

path = "F:/Masters/Dissertation/"

data = pd.read_csv(path+"Sample_Data.csv")

data["Date"] = pd.to_datetime(data["Date"],format="%d/%m/%Y")
data = data[["Date","SHEL.L"]]

data["Price_Return"] = (data["SHEL.L"].shift(1)/data["SHEL.L"])-1

plt.plot(data["Price_Return"])