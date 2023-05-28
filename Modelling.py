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
encoder = tf.keras.layers.LSTM(1,return_sequences=True)
encoder_outputs= encoder(encoder_inputs)
encoder_model = tf.keras.models.Model(encoder_inputs,encoder_outputs)



encoder_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(encoder_model.summary())

# Encoder
# encoder_inputs = tf.keras.Input(shape=(4,))
# encoder_embedding = tf.keras.layers.Embedding(vocab_size,embedding_dim)(encoder_inputs)





#%%



input_x = temp5.drop(['Actual Sales','Channel'],1)
input_x = input_x.to_numpy()
input_y = temp5['Actual Sales']

nrows = input_x.shape[0] - window_size + 1
p,q = input_x.shape
m,n = input_x.strides
strided = np.lib.stride_tricks.as_strided
out = strided(input_x,shape=(nrows,window_size,q),strides=(m,m,n))
input_y = input_y[window_size-1:]
input_y = np.array(input_y)

X_fut = out[[out.shape[0]-1],:,:]
actual_fut = actual_df.iloc[[len(actual_df)-1]]
actual_df = actual_df.iloc[:len(actual_df)-1]

input_y = input_y[:out.shape[0]-1]
out = out[:out.shape[0]-1,:,:]

cutoff = round(out.shape[0]*0.85)

X_train = out[:cutoff,:,:]
actual_train = actual_df.iloc[:cutoff]

X_test = out[cutoff:,:,:]
actual_test = actual_df.iloc[cutoff:]

Y_train = input_y[:cutoff]
Y_test = input_y[cutoff:]





num_units = 100
activation_function = "tanh"
optimizer = 'adam'
loss_function = 'msle'
num_epochs = 10000
window_size = 2














final_df = pd.DataFrame()

for ch in list(data.Channel.unique()): 
    temp = data[data["Channel"]==ch]
       
    temp_orig = temp.copy()
    
    scaler_x = preprocessing.MinMaxScaler(feature_range=(0,1))
    X = temp.drop(['Actual Sales'],axis=1)
    cols = list(X.columns)
    X = np.array(X).reshape((len(X),len(cols)))
    X_scaled = scaler_x.fit_transform(X)
    
    scaler_y = preprocessing.MinMaxScaler(feature_range=(0,1))
    Y = temp[["Actual Sales"]]
    Y = np.array(Y).reshape((len(Y),1))
    Y_scaled = scaler_y.fit_transform(Y)
    
    comb_df = np.concatenate((X_scaled,Y_scaled),axis=1)
    cols.append("Actual Sales")
    comb_df = pd.DataFrame(data=comb_df,columns=cols)
    
    X_train_comb = np.empty((0,window_size,len(cols)-2))
    X_test_comb = np.empty((0,window_size,len(cols)-2))
    X_fut_comb = np.empty((0,window_size,len(cols)-2))
    Y_train_comb = np.empty((0))
    Y_test_comb = np.empty((0))
    actual_train_comb = pd.DataFrame()
    actual_test_comb = pd.DataFrame()
    actual_fut_comb = pd.DataFrame()
    
    for cat,cat_2 in zip(list(comb_df.Category.unique()),list(temp_orig.Category.unique())):
        temp2 = comb_df[comb_df.Category==cat]
        temp2 = temp2.sort_values(by=["TYear","TWeek"])
        temp2 = temp2.reset_index(drop=True)
        
        temp3 = temp_orig[temp_orig.Category==cat_2]
        temp3 = temp3.sort_values(by=["TYear","TWeek"])
        temp3 = temp3.reset_index(drop=True)
    
        for sku,sku_2 in zip(list(temp2.GroupSKUName.unique()),list(temp3.GroupSKUName.unique())):
            
            temp5 = temp2[temp2.GroupSKUName==sku]
            temp5 = temp5.sort_values(by=["TYear","TWeek"])
            temp5 = temp5.reset_index(drop=True)
            
            if len(temp5)>=3:
            
                temp6 = temp3[temp3.GroupSKUName==sku_2]
                temp6 = temp6.sort_values(by=["TYear","TWeek"])
                temp6 = temp6.reset_index(drop=True)
                actual_df = temp6.iloc[window_size-1:]
                actual_df = actual_df.drop("Actual Sales",axis=1)
                
                input_x = temp5.drop(['Actual Sales','Channel'],1)
                input_x = input_x.to_numpy()
                input_y = temp5['Actual Sales']
                
                nrows = input_x.shape[0] - window_size + 1
                p,q = input_x.shape
                m,n = input_x.strides
                strided = np.lib.stride_tricks.as_strided
                out = strided(input_x,shape=(nrows,window_size,q),strides=(m,m,n))
                input_y = input_y[window_size-1:]
                input_y = np.array(input_y)
                
                X_fut = out[[out.shape[0]-1],:,:]
                actual_fut = actual_df.iloc[[len(actual_df)-1]]
                actual_df = actual_df.iloc[:len(actual_df)-1]
                
                input_y = input_y[:out.shape[0]-1]
                out = out[:out.shape[0]-1,:,:]
                
                cutoff = round(out.shape[0]*0.85)
                
                X_train = out[:cutoff,:,:]
                actual_train = actual_df.iloc[:cutoff]
                
                X_test = out[cutoff:,:,:]
                actual_test = actual_df.iloc[cutoff:]
                
                Y_train = input_y[:cutoff]
                Y_test = input_y[cutoff:]
                
                X_train_comb = np.concatenate((X_train_comb,X_train),axis=0)
                X_test_comb = np.concatenate((X_test_comb,X_test),axis=0)
                X_fut_comb = np.concatenate((X_fut_comb,X_fut),axis=0)
                Y_train_comb = np.concatenate((Y_train_comb,Y_train),axis=0)
                Y_test_comb = np.concatenate((Y_test_comb,Y_test),axis=0)
                
                actual_fut_comb = pd.concat([actual_fut_comb,actual_fut])
                actual_train_comb = pd.concat([actual_train_comb,actual_train])
                actual_test_comb = pd.concat([actual_test_comb,actual_test])
            
            else:
                continue
            
            
    if (X_train_comb.shape[0]>1) and (X_test_comb.shape[0]>1):
        
        print("##################Started##################")
        '''RNN and LSTM model'''
        
        bs = round(len(Y_train_comb)*0.10)
        if bs > 200:
            bs = 200
            
        # Initialize the RNN
        model_new = Sequential()
            
        # Adding the input layer and the LSTM layer
        model_new.add(LSTM(units = num_units, activation = activation_function,return_sequences=True,input_shape=(window_size,X_train_comb.shape[2])))
        model_new.add(LSTM(units = 50,activation = activation_function))
        model_new.add(Dropout(0.2))
        model_new.add(Dense(units = 1,activation="relu"))
        
        # Compiling the RNN
        model_new.compile(optimizer = optimizer, loss = loss_function, metrics=['msle'])
        
        # Early stopping and mode save
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
        mc = ModelCheckpoint('lstm_sh.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
        
        r = model_new.fit(X_train_comb, Y_train_comb, validation_data=(X_test_comb, Y_test_comb),
                          batch_size=bs,epochs = num_epochs,callbacks=[es,mc])
        best_iter = len(r.history["val_loss"])
        print(best_iter)
        clear_session()
        saved_model = load_model('lstm_sh.h5')
        
        train_pred =  saved_model.predict(X_train_comb,batch_size=1)
        train_pred = np.array(train_pred).reshape((len(train_pred),1))
        train_pred= scaler_y.inverse_transform(train_pred)
        
        test_pred =  saved_model.predict(X_test_comb,batch_size=1)
        test_pred = np.array(test_pred).reshape((len(test_pred),1))
        test_pred= scaler_y.inverse_transform(test_pred)
        
        fut_pred =  saved_model.predict(X_fut_comb,batch_size=1)
        fut_pred = np.array(fut_pred).reshape((len(fut_pred),1))
        fut_pred= scaler_y.inverse_transform(fut_pred)
        
        act_train = np.array(Y_train_comb).reshape((len(Y_train_comb),1))
        act_train = scaler_y.inverse_transform(act_train)
        
        act_test = np.array(Y_test_comb).reshape((len(Y_test_comb),1))
        act_test = scaler_y.inverse_transform(act_test)
        
        actual_train_comb["Projected Sales"] = train_pred
        actual_train_comb["Actual Sales"] = act_train
        actual_train_comb["Portion"] = "train"
        
        actual_test_comb["Projected Sales"] = test_pred
        actual_test_comb["Actual Sales"] = act_test
        actual_test_comb["Portion"] = "test"
        
        actual_fut_comb["Projected Sales"] = fut_pred
        actual_fut_comb["Actual Sales"] = np.nan
        actual_fut_comb["Portion"] = "future"
        
        final_df = pd.concat([final_df,actual_train_comb,actual_test_comb,actual_fut_comb])
