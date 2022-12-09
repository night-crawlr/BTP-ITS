
import time
st_lib = time.time()
import numpy as np
import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import os, psutil
import csv
from sklearn.metrics import mean_squared_error
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from bokeh.plotting import figure, output_file, show
et_lib = time.time()
elp_lib = et_lib - st_lib
print('Importing lib in:', elp_lib, ' sec')


st_code = time.time()
def load_csv(fir_dir, col, scenario):
    file_all = []
    
    if scenario == "freeway":
        file_all = ['1.csv','2.csv','3.csv','4.csv','5.csv','6.csv','7.csv','8.csv','9.csv',
        '10.csv', '11.csv','12.csv','13.csv']
    file_name = []
    for i in file_all:
        file_name.append(os.path.join(fir_dir, i))
    all_data = []
    for filename in file_name:
        csvfile = open(filename,'r')
        reader = csv.reader(csvfile)
        a = []
        for line in reader:
            a.append(line)
        b = []
        for i in range(len(a)):
            b.append(a[i][col])
        data = b[1:]
        all_data.extend(data)
        csvfile.close()
    data = np.array(all_data,dtype=float)
    return data


data1 = load_csv('./DATA/402214', 2, "freeway")
data2 = load_csv('./DATA/402835', 2, "freeway")
data3 = load_csv('./DATA/414025', 2, "freeway")
data4 = load_csv('./DATA/402510', 2, "freeway")

#print(len(data1))

scaler=MinMaxScaler(feature_range=(0,1))
d1=scaler.fit_transform(np.array(data1).reshape(-1,1))
d2=scaler.fit_transform(np.array(data2).reshape(-1,1))
d3=scaler.fit_transform(np.array(data3).reshape(-1,1))
d4=scaler.fit_transform(np.array(data4).reshape(-1,1))

#print(len(d1))
#print(len(d2))
#print(len(d3))
#print(len(d4))


def generate_data(data1, data2, data3, data4,  seq_len, pre_len, pre_sens_num):
    data = np.stack((d1[0:26191], d2[0:26191], d3[0:26191], d4[0:26191]), axis=1)
    x_data_cnn,x_data_lstm, x_w, x_d, label = load_data(data, seq_len ,pre_len, pre_sens_num)

    row = 2016  
    train_x_data = x_data_cnn[:-row]
    test_data = x_data_cnn[-row:]
    train_lstm = x_data_lstm[:-row]
    test_lstm = x_data_lstm[-row:]
    train_w = x_w[:-row]
    test_w = x_w[-row:]
    train_d = x_d[:-row]
    test_d = x_d[-row:]
    train_l = label[:-row]
    test_l = label[-row:]
    return train_x_data,train_lstm, train_w, train_d, train_l, \
           test_data, test_lstm, test_w, test_d, test_l


def load_data(data, seq_len=15, his=1, pre_sens_num=1):
    sequence_length = seq_len + his
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    result = np.stack(result, axis=0)
    train = result[:]
    x_train = train[:, :seq_len]
    x_lstm_train = train[:, :seq_len, pre_sens_num-1]
    x_wd_train = train[:, :seq_len, pre_sens_num-1]
    y_train = train[:, -1, pre_sens_num-1]
    x_data_cnn = []
    x_data_lstm = []
    x_w = []
    x_d = []
    label = []
    for i in range (len(train)):
        if i >= 2016:
            x_data_cnn.append(x_train[i])
            x_data_lstm.append(x_lstm_train[i])
            x_w.append(x_wd_train[i - 2016 + 8])
            x_d.append(x_wd_train[i - 288 + 8])
            label.append(y_train[i])
    x_data_cnn = np.array(x_data_cnn)
    x_data_lstm = np.array(x_data_lstm)
    x_w = np.array(x_w)
    x_d = np.array(x_d)
    label = np.array(label)
    return x_data_cnn,x_data_lstm,x_w,x_d,label

data = np.stack((d1[0:26191], d2[0:26191], d3[0:26191], d4[0:26191]), axis=1)

seq_len= 1
his=1
pre_sens_num=1
sequence_length = seq_len + his
result = []
for index in range(len(data) - sequence_length):
    result.append(data[index: index + sequence_length])
result = np.stack(result, axis=0)
train = result[:]
x_train = train[:, :seq_len]
x_lstm_train = train[:, :seq_len, pre_sens_num-1]
x_wd_train = train[:, :seq_len, pre_sens_num-1]
y_train = train[:, -1, pre_sens_num-1]
x_data_cnn = []
x_data_lstm = []
x_w = []
x_d = []
label = []
for i in range (len(train)):
    if i >= 2016:
        x_data_cnn.append(x_train[i])
        x_data_lstm.append(x_lstm_train[i])
        x_w.append(x_wd_train[i - 2016 + 8])
        x_d.append(x_wd_train[i - 288 + 8])
        label.append(y_train[i])
x_data_cnn = np.array(x_data_cnn)
x_data_lstm = np.array(x_data_lstm)
x_w = np.array(x_w)
x_d = np.array(x_d)
label = np.array(label)


epoch= 50
day = 288
week = 2016
seq_len = 1 
#1=5min, 3=15min, 6=30min, 12=60min 
pre_len = 12
#data 1-7
pre_sens_num = 1

#train,test
train_data, train_lstm,train_w, train_d,train_l, test_data, test_lstm, test_w, test_d,test_l\
	= generate_data(data1, data2, data3, data4, seq_len, pre_len, pre_sens_num)

train_data = np.reshape(train_data,(train_data.shape[0], train_data.shape[1], train_data.shape[2], 1))
train_w = np.reshape(train_w,(train_w.shape[0], train_w.shape[1], 1))
train_d = np.reshape(train_d,(train_d.shape[0], train_d.shape[1], 1))


test_data = np.reshape(test_data,(test_data.shape[0], test_data.shape[1], test_data.shape[2], 1))
test_d = np.reshape(test_d,(test_d.shape[0], test_d.shape[1], 1))
test_w = np.reshape(test_w,(test_w.shape[0], test_w.shape[1], 1))

data_lstm = train_data[:, :, :, 0]
for i in range(1, train_data.shape[3]):
    data_lstm = np.concatenate((data_lstm, train_data[:, :, :, i]), axis=1)
    
data_lstm_test = test_data[:, :, :, 0]
for i in range(1, test_data.shape[3]):
    data_lstm_test = np.concatenate((data_lstm_test, test_data[:, :, :, i]), axis=1)    
    
    
data = np.stack((d1[0:26191], d2[0:26191], d3[0:26191], d4[0:26191]), axis=1)

x_data_cnn,x_data_lstm, x_w, x_d, label = load_data(data, seq_len ,pre_len, pre_sens_num)


#print(data)


row = 2016
train_x_data = x_data_cnn[:-row]
test_data = x_data_cnn[-row:]
train_lstm = x_data_lstm[:-row]
test_lstm = x_data_lstm[-row:]
train_w = x_w[:-row]
test_w = x_w[-row:]
train_d = x_d[:-row]
test_d = x_d[-row:]
train_l = label[:-row]
test_l = label[-row:]


#print(x_data_cnn.shape)
#print(x_data_lstm.shape)
#print(x_w.shape)
#print(x_d.shape)
#print(label.shape)

#print(train_x_data.shape)
#print(test_data.shape)
#print(train_lstm.shape)
#print(test_lstm.shape)
#print(train_w.shape)
#print(test_w.shape)
#print(train_d.shape)
#print(test_d.shape)
#print(train_l.shape)
#print(test_l.shape)

#print(result.shape)
#print(train.shape)
#print(x_train.shape)
#print(x_lstm_train.shape)
#print(x_wd_train.shape)
#print(y_train.shape)

#print(data_lstm.shape)
#print(train_data.shape)
#print(data_lstm.shape)
#print(test_data.shape)

model = load_model('./cnn_lstm_periodicity/models_new/CNN-LSTMpcdw_new_5')

st_predict = time.time()
test_predict=model.predict([test_data,data_lstm_test,test_w,test_d])
stop_predict = time.time()
tt_predict = stop_predict - st_predict
print('Prediction Time: ', tt_predict, 'sec')
#print(test_predict.shape)
#print(test_predict[0:1])
test_predict2 = test_predict[:, -1]
#print(test_predict2[0:1])
#print(test_predict2.shape)


test_predict3 = scaler.inverse_transform(test_predict2)
#print(test_predict3[0:1])
#print(test_l.shape)
test_l = scaler.inverse_transform(test_l)
#print(test_l[0:3])


test_predict = test_predict3.flatten()
test_l = test_l.flatten()

def rmse_train(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def my_loss(y_true, y_pred):
    L1 = np.sum(np.abs(y_true - y_pred))
    L2 = np.sum(np.square(y_true - y_pred))
    mse = K.mean(K.square(y_true - y_pred), axis = -1)
    return L1 + L2 + mse

def predict_point_by_point(model, data):
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size, ))
    print ("predict_size:", predicted.size)
    return predicted

def MAE(pre,true):
    c = 0
    b = abs(np.subtract(pre, true))
    for i in range(len(b)):
        c = c + b[i]
    return c / len(b)

def MAPE(pre, true):
    a=0
    for i in range(len(pre)):
        x = (abs(pre[i] - true[i])/true[i])
        a = a + x
    return a / len(pre)

def RMSE(pre,true):
    c = 0
    b = abs(np.subtract(pre, true))
    b = b*b
    for i in range(len(b)):
        c = c + b[i]
    d = (c / len(b))**0.5
    return d



print ("MAE:", MAE(test_l,test_predict))
print ("MAPE:", MAPE(test_l,test_predict))
print ("RMSE:", RMSE(test_l,test_predict))
st_code_end = time.time()
elp_code = st_code_end - st_code
print('Code takes:', elp_code, ' sec to run')

process = psutil.Process(os.getpid())
print('Program takes',process.memory_info().rss / 1024 ** 2, 'mega-bytes of RAM')


"""
# Plot input data  and predicted output
len_day = 33
p = figure(x_axis_type = 'datetime',title="Traffic Flow", x_axis_label='Time', y_axis_label='Veh/5min',plot_width=1200, plot_height=600)

output_file("/home/pi/Desktop/Lovish/cnn_lstm_periodicity/CNN_LSTM_Traffic flow with Periodicity.html")

# # #For graphing using Bokeh
start = (int)(len_day - 6)
end = (int)(24*len_day)
timeAxis = [datetime.datetime(year=2021, month=8, day=9, hour=0, minute=0) + datetime.timedelta(minutes=5*i) for i in range(0,end)]
true = np.transpose(test_l)
p.line(timeAxis, test_l, color='green', legend_label='Expected Traffic Flow CNN-LSTM with Periodicity',line_width = 2)
p.line(timeAxis, test_predict, color='red', legend_label='Predicted Traffic Flow CNN-LSTM with Periodicity',line_width = 3)

# # show the results
show(p)
"""

print(test_predict.shape)
print(test_predict2.shape)
print(test_predict3.shape)
print(test_l.shape)


import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('./cnn_lstm_periodicity/models_new/CNN-LSTMpcdw_new_5') # path to the SavedModel directory

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter=True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
tf.lite.OpsSet.SELECT_TF_OPS]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

# Save the model.
with open('cnn_lstm_float16.tflite', 'wb') as f:
  f.write(tflite_model)

import numpy as np
import tensorflow as tf
import time
def interpret_tflite(PATH):
  # Load the TFLite model and allocate tensors.
  interpreter = tf.lite.Interpreter(model_path=PATH)
  interpreter.allocate_tensors()

  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  print(input_details)
  print(interpreter.get_signature_list())
  print(interpreter.get_signature_runner('serving_default'))
  interpreter.resize_tensor_input(0,[2016,1,4], strict=True)
  interpreter.resize_tensor_input(1,[2016,1,1], strict=True)
  interpreter.resize_tensor_input(2,[2016,1,4,1], strict=True)
  interpreter.resize_tensor_input(3,[2016,1,1], strict=True)

  # Test the model on random input data.
  predictions = []
  
  st = time.time()
  
  interpreter.allocate_tensors()
  interpreter.set_tensor(0,data_lstm_test.astype(np.float32))
  interpreter.set_tensor(1,test_d.astype(np.float32))
  interpreter.set_tensor(2,test_data.astype(np.float32))
  interpreter.set_tensor(3,test_w.astype(np.float32))

  interpreter.invoke()
  x1=interpreter.get_tensor(output_details[0]['index'])
  
  # for i in range(2016):
  #   interpreter.set_tensor(input_details[0]['index'], np.expand_dims(data_lstm_test[i].astype(np.float32),axis=0))
  #   interpreter.set_tensor(input_details[1]['index'], np.expand_dims(test_d[i].astype(np.float32),axis=0))
  #   interpreter.set_tensor(input_details[2]['index'], np.expand_dims(test_data[i].astype(np.float32),axis=0))
  #   interpreter.set_tensor(input_details[3]['index'], np.expand_dims(test_w[i].astype(np.float32),axis=0))
  #   interpreter.invoke()
  #   # The function `get_tensor()` returns a copy of the tensor data.
  #   # Use `tensor()` in order to get a pointer to the tensor.
  #   output_data = interpreter.get_tensor(output_details[0]['index'])
  #   predictions.append(scaler.inverse_transform(output_data[0]))
  print(x1.shape)
  predictions = scaler.inverse_transform(np.expand_dims(np.squeeze(x1),axis=1))

  
  et = time.time()
  
  # print(PATH,"Details : ")  
  # print("Time taken to predict : ",et-st," before qua : ",tt_predict)
  # print ("MAE:", MAE(test_l,(np.squeeze(np.array(predictions)))) , "Before : ",MAE(test_l,test_predict))
  # print ("MAPE:", MAPE(test_l,(np.squeeze(np.array(predictions)))),"Before : ",MAPE(test_l,test_predict))
  # print ("RMSE:", RMSE(test_l,(np.squeeze(np.array(predictions)))),"Before : ",RMSE(test_l,test_predict))
  
  print(PATH,"Details : ")  
  print("Time taken to predict : ",et-st)
  print ("MAE:", MAE(test_l,(np.squeeze(np.array(predictions)))) )
  print ("MAPE:", MAPE(test_l,(np.squeeze(np.array(predictions)))))
  print ("RMSE:", RMSE(test_l,(np.squeeze(np.array(predictions)))))
  

  process = psutil.Process(os.getpid())
  print(os.getpid())
  print('Program takes',process.memory_info().rss / 1024 ** 2, 'mega-bytes of RAM')
  print()
  print()

import numpy as np
import tensorflow as tf
import time
def interpret_tflite_by_sample(PATH):
  # Load the TFLite model and allocate tensors.
  interpreter = tf.lite.Interpreter(model_path=PATH)
  interpreter.allocate_tensors()

  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  print(input_details)
  print(interpreter.get_signature_list())
  print(interpreter.get_signature_runner('serving_default'))
  # interpreter.resize_tensor_input(0,[2016,1,4], strict=True)
  # interpreter.resize_tensor_input(1,[2016,1,1], strict=True)
  # interpreter.resize_tensor_input(2,[2016,1,4,1], strict=True)
  # interpreter.resize_tensor_input(3,[2016,1,1], strict=True)

  # Test the model on random input data.
  predictions = []
  
  st = time.time()
  
  interpreter.allocate_tensors()
  # interpreter.set_tensor(0,data_lstm_test.astype(np.float32))
  # interpreter.set_tensor(1,test_d.astype(np.float32) )
  # interpreter.set_tensor(2,test_data.astype(np.float32))
  # interpreter.set_tensor(3,test_w.astype(np.float32))

  # interpreter.invoke()
  # x1=interpreter.get_tensor(output_details[0]['index'])
  
  for i in range(2016):
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(data_lstm_test[i].astype(np.float32),axis=0))
    interpreter.set_tensor(input_details[1]['index'], np.expand_dims(test_d[i].astype(np.float32),axis=0))
    interpreter.set_tensor(input_details[2]['index'], np.expand_dims(test_data[i].astype(np.float32),axis=0))
    interpreter.set_tensor(input_details[3]['index'], np.expand_dims(test_w[i].astype(np.float32),axis=0))
    interpreter.invoke()
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predictions.append(scaler.inverse_transform(output_data[0]))
  #print(x1.shape)
  #predictions = scaler.inverse_transform(np.expand_dims(np.squeeze(x1),axis=1))

  
  et = time.time()
  
  # print(PATH,"Details : ")  
  # print("Time taken to predict : ",et-st," before qua : ",tt_predict)
  # print ("MAE:", MAE(test_l,(np.squeeze(np.array(predictions)))) , "Before : ",MAE(test_l,test_predict))
  # print ("MAPE:", MAPE(test_l,(np.squeeze(np.array(predictions)))),"Before : ",MAPE(test_l,test_predict))
  # print ("RMSE:", RMSE(test_l,(np.squeeze(np.array(predictions)))),"Before : ",RMSE(test_l,test_predict))
  
  print(PATH,"Details : ")  
  print("Time taken to predict : ",et-st)
  print ("MAE:", MAE(test_l,(np.squeeze(np.array(predictions)))) )
  print ("MAPE:", MAPE(test_l,(np.squeeze(np.array(predictions)))))
  print ("RMSE:", RMSE(test_l,(np.squeeze(np.array(predictions)))))
  

  process = psutil.Process(os.getpid())
  print(os.getpid())
  print('Program takes',process.memory_info().rss / 1024 ** 2, 'mega-bytes of RAM')
  print()
  print()

interpret_tflite('cnn_lstm_float16.tflite')
interpret_tflite_by_sample('cnn_lstm_float16.tflite')