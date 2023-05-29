#%%
# Importing the Libraries
import pandas as pd
import numpy as np
import tensorflow as tf

data_path = "M:/Dissertation/Data/"
#%%
# Reading the Data
stock_data = pd.read_csv(data_path+"Stocks_Data.csv")
stock_data["Datetime"] = pd.to_datetime(stock_data["Datetime"],format="%Y-%m-%d %H:00:00")

stock_data = stock_data.loc[stock_data.Stock=="BP.L"].reset_index(drop=True)
stock_data = stock_data.sort_values(by=["Datetime"])
stock_data = stock_data.drop("Stock",axis=1)

# Clipping the Targets
for col in ["O_S","H_S","L_S","C_S"]:
    max_thr = stock_data.loc[stock_data[col]>0,col].quantile(0.75)
    min_thr = stock_data.loc[stock_data[col]<0,col].quantile(0.75)
    stock_data[col] = stock_data[col].clip(min_thr,max_thr)
# %%
# Data Prep for RNN

# Reshaping the Data
window_size = 8
stock_data_x = stock_data[["O","H","L","C"]].to_numpy()
stock_data_y = stock_data[["O_S","H_S","L_S","C_S"]].to_numpy()

nrows = stock_data_x.shape[0] - window_size + 1
p,q = stock_data_x.shape
m,n = stock_data_x.strides
strided = np.lib.stride_tricks.as_strided
out = strided(stock_data_x,shape=(nrows,window_size,q),strides=(m,m,n))
stock_data_y = stock_data_y[window_size-1:]

# Splitting the Data
test_x = out[out.shape[0]-8:,:,:]

stock_data_y = stock_data_y[:out.shape[0]-8]
out = out[:out.shape[0]-8,:,:]

cutoff = round(out.shape[0]*0.95)

train_x = out[:cutoff,:,:]
valid_x = out[cutoff:,:,:]

train_y = stock_data_y[:cutoff]
valid_y = stock_data_y[cutoff:]

# Encoder-Decoder Architecture
encoder_inputs = tf.keras.Input(shape=(8,4))

lstm_1 = tf.keras.layers.LSTM(1,return_sequences=True,return_state=True)
lstm_1_outputs,s_h1,s_c1  = lstm_1(encoder_inputs)
s_h1 = tf.keras.layers.RepeatVector(16)(s_h1)
s_h1 = tf.keras.layers.Reshape((16,))(s_h1)
s_c1 = tf.keras.layers.RepeatVector(16)(s_c1)
s_c1 = tf.keras.layers.Reshape((16,))(s_c1)
lstm_1_states = [s_h1,s_c1]

lstm_2 = tf.keras.layers.LSTM(16,return_sequences=True,return_state=True)
lstm_2_outputs,s_h2,s_c2 = lstm_2(lstm_1_outputs,initial_state=lstm_1_states)
lstm_2_states = [s_h2,s_c2]

lstm_3 = tf.keras.layers.LSTM(16)
lstm_3_outputs = lstm_3(lstm_2_outputs,initial_state=lstm_2_states)

dense_1 = tf.keras.layers.Dense(4)
dense_1_outputs = dense_1(lstm_3_outputs)

model = tf.keras.models.Model(encoder_inputs,[lstm_1_outputs,lstm_2_outputs,lstm_3_outputs,dense_1_outputs])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())