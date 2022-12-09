{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "EwnWtnB58TVb",
        "outputId": "e1466c5c-afba-42d4-fade-f1e1b83fce34"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "/content/drive/MyDrive/Lovish\n"
          ]
        }
      ],
      "source": [
        "%cd /content/drive/MyDrive/Lovish"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 417
        },
        "id": "-cVfD4jO6o6w",
        "outputId": "07a6eb38-6f35-4fc2-a05b-da08cab070e7"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Importing lib in: 8.674377918243408  sec\n",
            "63/63 [==============================] - 3s 4ms/step\n",
            "Prediction Time:  3.370558977127075 sec\n",
            "[[43.37026 ]\n",
            " [45.545097]\n",
            " [39.021904]\n",
            " ...\n",
            " [48.23539 ]\n",
            " [49.559727]\n",
            " [52.856934]]\n",
            "[42. 40. 38. ... 44. 44. 42.]\n",
            "[43.37026  45.545097 39.021904 ... 48.23539  49.559727 52.856934]\n",
            "[42. 40. 38. ... 44. 44. 42.]\n",
            "MAE: 4.9151650709765295\n",
            "MAPE: 0.09779064263407072\n",
            "RMSE: 6.950996520508941\n",
            "Code takes: 36.42832064628601  sec to run\n",
            "Program takes 1053.62890625 mega-bytes of RAM\n"
          ]
        },
        {
          "data": {
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            },
            "text/plain": [
              "'\\n# Plot input data  and predicted output\\nlen_day = 33\\np = figure(x_axis_type = \\'datetime\\',title=\"Traffic Flow\", x_axis_label=\\'Time\\', y_axis_label=\\'Veh/5min\\',plot_width=1200, plot_height=600)\\n\\noutput_file(\"/home/pi/Desktop/Lovish/pcdwConvLSTM/pcdwConvLSTM_Traffic flow with Periodicity.html\")\\n\\n# # #For graphing using Bokeh\\nstart = (int)(len_day - 6)\\nend = (int)(24*len_day)\\ntimeAxis = [datetime.datetime(year=2021, month=8, day=9, hour=0, minute=0) + datetime.timedelta(minutes=5*i) for i in range(0,end)]\\ntrue = np.transpose(l_real)\\npred1 = l_real\\npred2 = list(pred1)\\n\\np.line(timeAxis, l_real, color=\\'green\\', legend_label=\\'Expected Traffic Flow ConvLSTM with Periodicity\\',line_width = 2)\\np.line(timeAxis, p_real, color=\\'red\\', legend_label=\\'Predicted Traffic Flow ConvLSTM with Periodicity\\',line_width = 3)\\n\\n# # show the results\\nshow(p)\\n'"
            ]
          },
          "execution_count": 2,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "import time\n",
        "st_lib = time.time()\n",
        "import numpy as np\n",
        "import datetime\n",
        "import tensorflow as tf\n",
        "from tensorflow.keras.models import load_model\n",
        "import pandas as pd\n",
        "import os, psutil\n",
        "import csv\n",
        "from sklearn.metrics import mean_squared_error\n",
        "import math\n",
        "from bokeh.plotting import figure, output_file, show\n",
        "et_lib = time.time()\n",
        "elp_lib = et_lib - st_lib\n",
        "print('Importing lib in:', elp_lib, ' sec')\n",
        "\n",
        "\n",
        "st_code = time.time()\n",
        "def load_data(data, seq_len=13, his=1, pre_sens_num=1):\n",
        "    max = np.max(data)\n",
        "    min = np.min(data)\n",
        "    med = max - min\n",
        "    data = np.array(data, dtype=float)\n",
        "    data_nor = (data - min) / med\n",
        "\n",
        "    sequence_length = seq_len + his\n",
        "    result = []\n",
        "    for index in range(len(data_nor) - sequence_length):\n",
        "        result.append(data_nor[index: index + sequence_length])\n",
        "    result = np.stack(result, axis=0)\n",
        "    train = result[:]\n",
        "    x_train = train[:, :seq_len]\n",
        "    x_lstm_train = train[:, :seq_len, pre_sens_num-1]\n",
        "    x_wd_train = train[:, :seq_len, pre_sens_num-1]\n",
        "    y_train = train[:, -6, pre_sens_num-1]\n",
        "    x_data_cnn = []\n",
        "    x_w = []\n",
        "    x_d = []\n",
        "    label = []\n",
        "    for i in range (len(train)):\n",
        "        if i >= 2016:\n",
        "            x_data_cnn.append(x_train[i])\n",
        "            x_w.append(x_wd_train[i - 2016 + 8])\n",
        "            x_d.append(x_wd_train[i - 288 + 8])\n",
        "            label.append(y_train[i])\n",
        "    x_data_cnn = np.array(x_data_cnn)\n",
        "    x_w = np.array(x_w)\n",
        "    x_d = np.array(x_d)\n",
        "    label = np.array(label)\n",
        "    return x_data_cnn,x_w,x_d,label,med,min\n",
        "\n",
        "\n",
        "def generate_data(data1, data2, data3, data4, seq_len, pre_len, pre_sens_num):\n",
        "    data = np.stack((data1[0:26191], data2[0:26191], data3[0:26191], data4[0:26191]), axis=1)\n",
        "    x_data_cnn,x_w, x_d, label, med, min = load_data(data, seq_len ,pre_len, pre_sens_num)\n",
        "\n",
        "    row = 2016\n",
        "    train_x_data = x_data_cnn[:-row]\n",
        "    test_data = x_data_cnn[-row:]\n",
        "    train_w = x_w[:-row]\n",
        "    test_w = x_w[-row:]\n",
        "    train_d = x_d[:-row]\n",
        "    test_d = x_d[-row:]\n",
        "    train_l = label[:-row]\n",
        "    test_l = label[-row:]\n",
        "    return train_x_data,train_w, train_d, train_l, \\\n",
        "           test_data, test_w, test_d, test_l, med, min\n",
        "\n",
        "def load_csv(fir_dir, col, scenario):\n",
        "    file_all = []\n",
        "    \n",
        "    if scenario == \"freeway\":\n",
        "        file_all = ['1.csv','2.csv','3.csv','4.csv','5.csv','6.csv','7.csv','8.csv','9.csv',\n",
        "        '10.csv', '11.csv','12.csv','13.csv']\n",
        "    file_name = []\n",
        "    for i in file_all:\n",
        "        file_name.append(os.path.join(fir_dir, i))\n",
        "    all_data = []\n",
        "    for filename in file_name:\n",
        "        csvfile = open(filename,'r')\n",
        "        reader = csv.reader(csvfile)\n",
        "        a = []\n",
        "        for line in reader:\n",
        "            a.append(line)\n",
        "        b = []\n",
        "        for i in range(len(a)):\n",
        "            b.append(a[i][col])\n",
        "        data = b[1:]\n",
        "        all_data.extend(data)\n",
        "        csvfile.close()\n",
        "    data = np.array(all_data,dtype=float)\n",
        "    return data\n",
        "\n",
        "\n",
        "data1 = load_csv('./DATA/402214', 2, \"freeway\")\n",
        "data2 = load_csv('./DATA/402835', 2, \"freeway\")\n",
        "data3 = load_csv('./DATA/414025', 2, \"freeway\")\n",
        "data4 = load_csv('./DATA/402510', 2, \"freeway\")\n",
        "\n",
        "#print(len(data1))\n",
        "\n",
        "epoch= 50\n",
        "day = 288\n",
        "week = 2016\n",
        "seq_len = 1\n",
        "#1=5min, 3=15min, 6=30min, 12=60min\n",
        "pre_len = 12\n",
        "#data 1-7\n",
        "pre_sens_num = 1\n",
        "\n",
        "#train,test\n",
        "train_data, train_w, train_d,train_l, test_data,test_w, test_d,test_l, test_med, test_min\\\n",
        "\t= generate_data(data1, data2, data3, data4, seq_len, pre_len, pre_sens_num)\n",
        "\n",
        "train_data = np.reshape(train_data,(train_data.shape[0], train_data.shape[1], train_data.shape[2], 1))\n",
        "train_w = np.reshape(train_w,(train_w.shape[0], train_w.shape[1], 1))\n",
        "train_d = np.reshape(train_d,(train_d.shape[0], train_d.shape[1], 1))\n",
        "\n",
        "\n",
        "test_data = np.reshape(test_data,(test_data.shape[0], test_data.shape[1], test_data.shape[2], 1))\n",
        "test_d = np.reshape(test_d,(test_d.shape[0], test_d.shape[1], 1))\n",
        "test_w = np.reshape(test_w,(test_w.shape[0], test_w.shape[1], 1))\n",
        "\n",
        "#print(test_data.shape)\n",
        "\n",
        "model = load_model('./pcdwConvLSTM/models_new/ConvLSTMpcdw_new_5')\n",
        "\n",
        "st_predict = time.time()\n",
        "test_predict=model.predict([test_data,test_w,test_d])\n",
        "stop_predict = time.time()\n",
        "tt_predict = stop_predict - st_predict\n",
        "print('Prediction Time: ', tt_predict, 'sec')\n",
        "\n",
        "#print(test_predict.shape)\n",
        "\n",
        "p_real = []\n",
        "l_real = []\n",
        "row=2016\n",
        "\n",
        "\n",
        "for i in range(row):\n",
        "    p_real.append(test_predict[i] * test_med + test_min)\n",
        "    l_real.append(test_l[i] * test_med + test_min)\n",
        "p_real = np.array(p_real)\n",
        "l_real = np.array(l_real)\n",
        "\n",
        "print(p_real)\n",
        "print(l_real)\n",
        "\n",
        "l_real[0:3]\n",
        "p_real.shape\n",
        "p_real=p_real[:,-1]\n",
        "p_real[0:3]\n",
        "#newarr = arr.reshape(arr.shape[0], (arr.shape[1]*arr.shape[2]))                        \n",
        "p_real.shape\n",
        "l_real = l_real.flatten()\n",
        "p_real = p_real.flatten()\n",
        "\n",
        "print(p_real)\n",
        "print(l_real)\n",
        "\n",
        "def rmse_train(y_true, y_pred):\n",
        "    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))\n",
        "\n",
        "def my_loss(y_true, y_pred):\n",
        "    L1 = np.sum(np.abs(y_true - y_pred))\n",
        "    L2 = np.sum(np.square(y_true - y_pred))\n",
        "    mse = K.mean(K.square(y_true - y_pred), axis = -1)\n",
        "    return L1 + L2 + mse\n",
        "\n",
        "def predict_point_by_point(model, data):\n",
        "    predicted = model.predict(data)\n",
        "    predicted = np.reshape(predicted, (predicted.size, ))\n",
        "    print (\"predict_size:\", predicted.size)\n",
        "    return predicted\n",
        "\n",
        "def MAE(pre,true):\n",
        "    c = 0\n",
        "    b = abs(np.subtract(pre, true))\n",
        "    for i in range(len(b)):\n",
        "        c = c + b[i]\n",
        "    return c / len(b)\n",
        "\n",
        "def MAPE(pre, true):\n",
        "    a=0\n",
        "    for i in range(len(pre)):\n",
        "        x = (abs(pre[i] - true[i])/true[i])\n",
        "        a = a + x\n",
        "    return a / len(pre)\n",
        "\n",
        "def RMSE(pre,true):\n",
        "    c = 0\n",
        "    b = abs(np.subtract(pre, true))\n",
        "    b = b*b\n",
        "    for i in range(len(b)):\n",
        "        c = c + b[i]\n",
        "    d = (c / len(b))**0.5\n",
        "    return d\n",
        "\n",
        "\n",
        "print (\"MAE:\", MAE(p_real, l_real))\n",
        "print (\"MAPE:\", MAPE(p_real, l_real))\n",
        "print (\"RMSE:\", RMSE(p_real, l_real))\n",
        "st_code_end = time.time()\n",
        "elp_code = st_code_end - st_code\n",
        "print('Code takes:', elp_code, ' sec to run')\n",
        "\n",
        "process = psutil.Process(os.getpid())\n",
        "print('Program takes',process.memory_info().rss / 1024 ** 2, 'mega-bytes of RAM')\n",
        "\n",
        "\n",
        "\"\"\"\n",
        "# Plot input data  and predicted output\n",
        "len_day = 33\n",
        "p = figure(x_axis_type = 'datetime',title=\"Traffic Flow\", x_axis_label='Time', y_axis_label='Veh/5min',plot_width=1200, plot_height=600)\n",
        "\n",
        "output_file(\"/home/pi/Desktop/Lovish/pcdwConvLSTM/pcdwConvLSTM_Traffic flow with Periodicity.html\")\n",
        "\n",
        "# # #For graphing using Bokeh\n",
        "start = (int)(len_day - 6)\n",
        "end = (int)(24*len_day)\n",
        "timeAxis = [datetime.datetime(year=2021, month=8, day=9, hour=0, minute=0) + datetime.timedelta(minutes=5*i) for i in range(0,end)]\n",
        "true = np.transpose(l_real)\n",
        "pred1 = l_real\n",
        "pred2 = list(pred1)\n",
        "\n",
        "p.line(timeAxis, l_real, color='green', legend_label='Expected Traffic Flow ConvLSTM with Periodicity',line_width = 2)\n",
        "p.line(timeAxis, p_real, color='red', legend_label='Predicted Traffic Flow ConvLSTM with Periodicity',line_width = 3)\n",
        "\n",
        "# # show the results\n",
        "show(p)\n",
        "\"\"\"\n",
        "\n",
        "\n",
        "    "
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pI-myh81-qcE",
        "outputId": "133352a2-4124-4ead-b70f-d86d2798b4b7"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "(2016, 2016, (2016, 1, 1), (2016, 1, 1), (2016, 1, 4, 1))"
            ]
          },
          "execution_count": 3,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "len(p_real),len(l_real), test_w.shape, test_d.shape, test_data.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cWCdzKuZsgOx",
        "outputId": "9250afea-fd34-4b15-887f-8f528e88a9d3"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "((22146, 1, 4, 1), (22146, 1, 1), (22146, 1, 1), (22146,))"
            ]
          },
          "execution_count": 4,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "train_data.shape,  train_w.shape, train_d.shape, train_l.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "54osMNWz9brd",
        "outputId": "b03b9b27-29f5-4666-c255-a64aceafb9c0"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Model: \"model_3\"\n",
            "__________________________________________________________________________________________________\n",
            " Layer (type)                   Output Shape         Param #     Connected to                     \n",
            "==================================================================================================\n",
            " main_input (InputLayer)        [(None, 1, 4, 1)]    0           []                               \n",
            "                                                                                                  \n",
            " time_distributed_15 (TimeDistr  (None, 1, 4, 26)    156         ['main_input[0][0]']             \n",
            " ibuted)                                                                                          \n",
            "                                                                                                  \n",
            " time_distributed_16 (TimeDistr  (None, 1, 4, 30)    2370        ['time_distributed_15[0][0]']    \n",
            " ibuted)                                                                                          \n",
            "                                                                                                  \n",
            " time_distributed_17 (TimeDistr  (None, 1, 4, 11)    1001        ['time_distributed_16[0][0]']    \n",
            " ibuted)                                                                                          \n",
            "                                                                                                  \n",
            " time_distributed_18 (TimeDistr  (None, 1, 44)       0           ['time_distributed_17[0][0]']    \n",
            " ibuted)                                                                                          \n",
            "                                                                                                  \n",
            " time_distributed_19 (TimeDistr  (None, 1, 30)       1350        ['time_distributed_18[0][0]']    \n",
            " ibuted)                                                                                          \n",
            "                                                                                                  \n",
            " auxiliary_input_w (InputLayer)  [(None, 1, 1)]      0           []                               \n",
            "                                                                                                  \n",
            " auxiliary_input_d (InputLayer)  [(None, 1, 1)]      0           []                               \n",
            "                                                                                                  \n",
            " lstm_18 (LSTM)                 (None, 1, 14)        2520        ['time_distributed_19[0][0]']    \n",
            "                                                                                                  \n",
            " lstm_20 (LSTM)                 (None, 1, 29)        3596        ['auxiliary_input_w[0][0]']      \n",
            "                                                                                                  \n",
            " lstm_22 (LSTM)                 (None, 1, 24)        2496        ['auxiliary_input_d[0][0]']      \n",
            "                                                                                                  \n",
            " lstm_19 (LSTM)                 (None, 14)           1624        ['lstm_18[0][0]']                \n",
            "                                                                                                  \n",
            " lstm_21 (LSTM)                 (None, 29)           6844        ['lstm_20[0][0]']                \n",
            "                                                                                                  \n",
            " lstm_23 (LSTM)                 (None, 24)           4704        ['lstm_22[0][0]']                \n",
            "                                                                                                  \n",
            " concatenate_3 (Concatenate)    (None, 67)           0           ['lstm_19[0][0]',                \n",
            "                                                                  'lstm_21[0][0]',                \n",
            "                                                                  'lstm_23[0][0]']                \n",
            "                                                                                                  \n",
            " dense_10 (Dense)               (None, 20)           1360        ['concatenate_3[0][0]']          \n",
            "                                                                                                  \n",
            " dense_11 (Dense)               (None, 10)           210         ['dense_10[0][0]']               \n",
            "                                                                                                  \n",
            " main_output (Dense)            (None, 1)            11          ['dense_11[0][0]']               \n",
            "                                                                                                  \n",
            "==================================================================================================\n",
            "Total params: 28,242\n",
            "Trainable params: 28,242\n",
            "Non-trainable params: 0\n",
            "__________________________________________________________________________________________________\n"
          ]
        }
      ],
      "source": [
        "model.summary()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "lMYDMfDCxhIp"
      },
      "outputs": [],
      "source": [
        "def scale(scores,test_med, test_min):\n",
        "  scaled=[]\n",
        "  for e in scores:\n",
        "    scaled.append(e*test_med + test_min)\n",
        "  return scaled"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "gHWpvIW6phZA"
      },
      "outputs": [],
      "source": [
        "finetuning_ratio = 0.9\n",
        "train_data_subset = train_data[:int(finetuning_ratio*len(train_data))] # out of 22146 training samples\n",
        "train_w_subset = train_w[:int(finetuning_ratio*len(train_data))] # out of 22146 training samples\n",
        "train_d_subset = train_d[:int(finetuning_ratio*len(train_data))] # out of 22146 training samples\n",
        "\n",
        "train_l_subset = train_l[:int(finetuning_ratio*len(train_data))] "
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WpgHAgauxjKG",
        "outputId": "c15efe64-75b6-4ab1-8d93-ec1af476950d"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "((19931, 1, 4, 1), (19931, 1, 1), (19931, 1, 1), (19931,))"
            ]
          },
          "execution_count": 8,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "train_data_subset.shape,train_w_subset.shape,train_d_subset.shape,np.array(scale(train_l_subset,test_med,test_min)).shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0WE42qBDx-JV"
      },
      "outputs": [],
      "source": [
        "scaled_l_subset = np.array(scale(train_l_subset,test_med,test_min))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Ku5uVFpYymfR",
        "outputId": "517a41b9-aefa-495c-bcd3-aedb7c168c3e"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/\n",
            "Collecting tensorflow-model-optimization\n",
            "  Downloading tensorflow_model_optimization-0.7.3-py2.py3-none-any.whl (238 kB)\n",
            "\u001b[K     |████████████████████████████████| 238 kB 5.3 MB/s \n",
            "\u001b[?25hRequirement already satisfied: six~=1.10 in /usr/local/lib/python3.8/dist-packages (from tensorflow-model-optimization) (1.15.0)\n",
            "Requirement already satisfied: numpy~=1.14 in /usr/local/lib/python3.8/dist-packages (from tensorflow-model-optimization) (1.21.6)\n",
            "Requirement already satisfied: dm-tree~=0.1.1 in /usr/local/lib/python3.8/dist-packages (from tensorflow-model-optimization) (0.1.7)\n",
            "Installing collected packages: tensorflow-model-optimization\n",
            "Successfully installed tensorflow-model-optimization-0.7.3\n"
          ]
        }
      ],
      "source": [
        "!pip install tensorflow-model-optimization\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0cRwrpaNyH4E"
      },
      "outputs": [],
      "source": [
        "# MAKING A QUANT AWARE MODEL\n",
        "import tensorflow_model_optimization as tfmot\n",
        "\n",
        "quantize_model = tfmot.quantization.keras.quantize_model"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5m5aCSUvEywS",
        "outputId": "01b3f318-a9b7-4e1f-cd9b-84901e29f68e"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "(<keras.optimizers.optimizer_v2.adam.Adam at 0x7fda1ee5dfa0>,\n",
              " <function keras.losses.mean_squared_error(y_true, y_pred)>,\n",
              " [<keras.metrics.base_metric.Mean at 0x7fda2017ce20>],\n",
              " ['loss'])"
            ]
          },
          "execution_count": 12,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "model.optimizer , model.loss, model.metrics, model.metrics_names"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "JVb_2-9rgmW1",
        "outputId": "c9f552fc-30d2-4b40-d979-b7257a1b15da"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Model: \"model_3\"\n",
            "__________________________________________________________________________________________________\n",
            " Layer (type)                   Output Shape         Param #     Connected to                     \n",
            "==================================================================================================\n",
            " main_input (InputLayer)        [(None, 1, 4, 1)]    0           []                               \n",
            "                                                                                                  \n",
            " time_distributed_15 (TimeDistr  (None, 1, 4, 26)    156         ['main_input[0][0]']             \n",
            " ibuted)                                                                                          \n",
            "                                                                                                  \n",
            " time_distributed_16 (TimeDistr  (None, 1, 4, 30)    2370        ['time_distributed_15[0][0]']    \n",
            " ibuted)                                                                                          \n",
            "                                                                                                  \n",
            " time_distributed_17 (TimeDistr  (None, 1, 4, 11)    1001        ['time_distributed_16[0][0]']    \n",
            " ibuted)                                                                                          \n",
            "                                                                                                  \n",
            " time_distributed_18 (TimeDistr  (None, 1, 44)       0           ['time_distributed_17[0][0]']    \n",
            " ibuted)                                                                                          \n",
            "                                                                                                  \n",
            " time_distributed_19 (TimeDistr  (None, 1, 30)       1350        ['time_distributed_18[0][0]']    \n",
            " ibuted)                                                                                          \n",
            "                                                                                                  \n",
            " auxiliary_input_w (InputLayer)  [(None, 1, 1)]      0           []                               \n",
            "                                                                                                  \n",
            " auxiliary_input_d (InputLayer)  [(None, 1, 1)]      0           []                               \n",
            "                                                                                                  \n",
            " lstm_18 (LSTM)                 (None, 1, 14)        2520        ['time_distributed_19[0][0]']    \n",
            "                                                                                                  \n",
            " lstm_20 (LSTM)                 (None, 1, 29)        3596        ['auxiliary_input_w[0][0]']      \n",
            "                                                                                                  \n",
            " lstm_22 (LSTM)                 (None, 1, 24)        2496        ['auxiliary_input_d[0][0]']      \n",
            "                                                                                                  \n",
            " lstm_19 (LSTM)                 (None, 14)           1624        ['lstm_18[0][0]']                \n",
            "                                                                                                  \n",
            " lstm_21 (LSTM)                 (None, 29)           6844        ['lstm_20[0][0]']                \n",
            "                                                                                                  \n",
            " lstm_23 (LSTM)                 (None, 24)           4704        ['lstm_22[0][0]']                \n",
            "                                                                                                  \n",
            " concatenate_3 (Concatenate)    (None, 67)           0           ['lstm_19[0][0]',                \n",
            "                                                                  'lstm_21[0][0]',                \n",
            "                                                                  'lstm_23[0][0]']                \n",
            "                                                                                                  \n",
            " dense_10 (Dense)               (None, 20)           1360        ['concatenate_3[0][0]']          \n",
            "                                                                                                  \n",
            " dense_11 (Dense)               (None, 10)           210         ['dense_10[0][0]']               \n",
            "                                                                                                  \n",
            " main_output (Dense)            (None, 1)            11          ['dense_11[0][0]']               \n",
            "                                                                                                  \n",
            "==================================================================================================\n",
            "Total params: 28,242\n",
            "Trainable params: 28,242\n",
            "Non-trainable params: 0\n",
            "__________________________________________________________________________________________________\n"
          ]
        }
      ],
      "source": [
        "model.summary()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "6a03MnUblpc7"
      },
      "outputs": [],
      "source": [
        "def apply_quantization_to_dense(layer):\n",
        "  if isinstance(layer, tf.keras.layers.Dense):\n",
        "    return tfmot.quantization.keras.quantize_annotate_layer(layer)\n",
        "  return layer\n",
        "\n",
        "# Use `tf.keras.models.clone_model` to apply `apply_quantization_to_dense` \n",
        "# to the layers of the model.\n",
        "annotated_model = tf.keras.models.clone_model(\n",
        "    model,\n",
        "    clone_function=apply_quantization_to_dense,\n",
        ")\n",
        "\n",
        "# Now that the Dense layers are annotated,\n",
        "# `quantize_apply` actually makes the model quantization aware.\n",
        "quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)\n",
        "quant_aware_model.compile(optimizer='Adam',loss=tf.keras.losses.MeanSquaredError())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true,
          "base_uri": "https://localhost:8080/"
        },
        "id": "7wXrLRTekYyx",
        "outputId": "bd76eafd-284f-43ab-eff0-9efdf9d0225a"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Epoch 1/25\n",
            "499/499 [==============================] - 37s 39ms/step - loss: 0.0177 - val_loss: 0.0055\n",
            "Epoch 2/25\n",
            "499/499 [==============================] - 6s 11ms/step - loss: 0.0054 - val_loss: 0.0039\n",
            "Epoch 3/25\n",
            "499/499 [==============================] - 6s 12ms/step - loss: 0.0043 - val_loss: 0.0033\n",
            "Epoch 4/25\n",
            "499/499 [==============================] - 6s 11ms/step - loss: 0.0039 - val_loss: 0.0029\n",
            "Epoch 5/25\n",
            "499/499 [==============================] - 6s 11ms/step - loss: 0.0036 - val_loss: 0.0029\n",
            "Epoch 6/25\n",
            "499/499 [==============================] - 6s 12ms/step - loss: 0.0035 - val_loss: 0.0026\n",
            "Epoch 7/25\n",
            "499/499 [==============================] - 6s 11ms/step - loss: 0.0034 - val_loss: 0.0025\n",
            "Epoch 8/25\n",
            "499/499 [==============================] - 5s 11ms/step - loss: 0.0034 - val_loss: 0.0026\n",
            "Epoch 9/25\n",
            "499/499 [==============================] - 5s 11ms/step - loss: 0.0033 - val_loss: 0.0084\n",
            "Epoch 10/25\n",
            "499/499 [==============================] - 6s 12ms/step - loss: 0.0035 - val_loss: 0.0025\n",
            "Epoch 11/25\n",
            "499/499 [==============================] - 6s 12ms/step - loss: 0.0032 - val_loss: 0.0024\n",
            "Epoch 12/25\n",
            "499/499 [==============================] - 6s 12ms/step - loss: 0.0034 - val_loss: 0.0023\n",
            "Epoch 13/25\n",
            "499/499 [==============================] - 6s 12ms/step - loss: 0.0032 - val_loss: 0.0041\n",
            "Epoch 14/25\n",
            "499/499 [==============================] - 6s 12ms/step - loss: 0.0033 - val_loss: 0.0026\n",
            "Epoch 15/25\n",
            "499/499 [==============================] - 6s 12ms/step - loss: 0.0032 - val_loss: 0.0026\n",
            "Epoch 16/25\n",
            "499/499 [==============================] - 6s 11ms/step - loss: 0.0032 - val_loss: 0.0022\n",
            "Epoch 17/25\n",
            "499/499 [==============================] - 5s 11ms/step - loss: 0.0031 - val_loss: 0.0023\n",
            "Epoch 18/25\n",
            "499/499 [==============================] - 6s 12ms/step - loss: 0.0032 - val_loss: 0.0026\n",
            "Epoch 19/25\n",
            "499/499 [==============================] - 7s 14ms/step - loss: 0.0033 - val_loss: 0.0036\n",
            "Epoch 20/25\n",
            "499/499 [==============================] - 5s 11ms/step - loss: 0.0031 - val_loss: 0.0043\n",
            "Epoch 21/25\n",
            "499/499 [==============================] - 6s 11ms/step - loss: 0.0032 - val_loss: 0.0021\n",
            "Epoch 22/25\n",
            "499/499 [==============================] - 6s 12ms/step - loss: 0.0032 - val_loss: 0.0023\n",
            "Epoch 23/25\n",
            "499/499 [==============================] - 6s 11ms/step - loss: 0.0030 - val_loss: 0.0021\n",
            "Epoch 24/25\n",
            "499/499 [==============================] - 8s 15ms/step - loss: 0.0030 - val_loss: 0.0022\n",
            "Epoch 25/25\n",
            "499/499 [==============================] - 6s 11ms/step - loss: 0.0031 - val_loss: 0.0020\n"
          ]
        }
      ],
      "source": [
        "his = quant_aware_model.fit([train_data_subset,train_w_subset,train_d_subset],train_l_subset,epochs=25,validation_split=0.2)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "Flg_qpe4QiLf",
        "outputId": "8d4b8255-1fa0-4f09-fa0e-ba57a94513d7"
      },
      "outputs": [
        {
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "from matplotlib import pyplot as plt\n",
        "plt.plot(his.history['loss'])\n",
        "plt.plot(his.history['val_loss'])\n",
        "plt.legend(['train','val'])\n",
        "plt.show()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "mPCjlmU3mDcI",
        "outputId": "a84b3dab-1c05-4d18-cc0c-6b54129a6d7f"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "63/63 [==============================] - 0s 4ms/step\n",
            "63/63 [==============================] - 3s 4ms/step\n"
          ]
        }
      ],
      "source": [
        "test_predict=model.predict([test_data,test_w,test_d])\n",
        "test_predict_quant = quant_aware_model.predict([test_data,test_w,test_d])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "ZDm0PdAzSVZA",
        "outputId": "761de1b5-add7-457a-e353-cd19fbb12132"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "(2016, 1, 4, 1)"
            ]
          },
          "execution_count": 18,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "test_data.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "pJL_ucdgmPd2"
      },
      "outputs": [],
      "source": [
        "predicted = scale(test_predict[:,-1],test_med, test_min)\n",
        "predicted_quant = scale(test_predict_quant[:,-1],test_med, test_min)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "a3-cw_a4m5sy",
        "outputId": "7d5fc0b5-b499-461c-e32f-944e06fa99aa"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "MAE: 4.915164953378576\n",
            "MAPE: 0.09779064232173779\n",
            "RMSE: 6.9509964941885825\n",
            "(Quant) MAE: 4.427303391737893\n",
            "(Quant) MAPE: 0.08860656593317949\n",
            "(Quant) RMSE: 6.214159530510585\n"
          ]
        }
      ],
      "source": [
        "print (\"MAE:\", MAE(predicted, l_real))\n",
        "print (\"MAPE:\", MAPE(predicted, l_real))\n",
        "print (\"RMSE:\", RMSE(predicted, l_real))\n",
        "\n",
        "print (\"(Quant) MAE:\", MAE(predicted_quant, l_real))\n",
        "print (\"(Quant) MAPE:\", MAPE(predicted_quant, l_real))\n",
        "print (\"(Quant) RMSE:\", RMSE(predicted_quant, l_real))\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "MxBKP-rPl6Te",
        "outputId": "113bdbdb-4241-4d10-c663-ab240e8723f0"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, lstm_cell_12_layer_call_fn, lstm_cell_12_layer_call_and_return_conditional_losses while saving (showing 5 of 23). These functions will not be directly callable after loading.\n",
            "/usr/local/lib/python3.8/dist-packages/tensorflow/lite/python/convert.py:766: UserWarning: Statistics for quantized inputs were expected, but not specified; continuing anyway.\n",
            "  warnings.warn(\"Statistics for quantized inputs were expected, but not \"\n"
          ]
        },
        {
          "data": {
            "text/plain": [
              "161768"
            ]
          },
          "execution_count": 21,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)\n",
        "converter.target_spec.supported_ops = [\n",
        "  tf.lite.OpsSet.TFLITE_BUILTINS, \n",
        "  tf.lite.OpsSet.SELECT_TF_OPS \n",
        "]\n",
        "converter.optimizations = [tf.lite.Optimize.DEFAULT]\n",
        "converter.target_spec.supported_types = [tf.float16]\n",
        "\n",
        "\n",
        "tflite_model = converter.convert()\n",
        "\n",
        "open(\"conv_lstm_model_QA.tflite\", \"wb\").write(tflite_model)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "3O6WhSuw7swa"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import tensorflow as tf\n",
        "import time\n",
        "def interpret_tflite(PATH):\n",
        "  # Load the TFLite model and allocate tensors.\n",
        "  interpreter = tf.lite.Interpreter(model_path=PATH)\n",
        "  interpreter.allocate_tensors()\n",
        "\n",
        "  # Get input and output tensors.\n",
        "  input_details = interpreter.get_input_details()\n",
        "  output_details = interpreter.get_output_details()\n",
        "\n",
        "\n",
        "  print((input_details))\n",
        "\n",
        "  # Test the model on random input data.\n",
        "  predictions = []\n",
        "  \n",
        "  st = time.time()\n",
        "  \n",
        "  for i in range(2016):\n",
        "    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(test_d[i].astype(np.float32),axis=0))\n",
        "    interpreter.set_tensor(input_details[1]['index'], np.expand_dims(test_data[i].astype(np.float32),axis=0))\n",
        "    interpreter.set_tensor(input_details[2]['index'], np.expand_dims(test_w[i].astype(np.float32),axis=0))\n",
        "    interpreter.invoke()\n",
        "    # The function `get_tensor()` returns a copy of the tensor data.\n",
        "    # Use `tensor()` in order to get a pointer to the tensor.\n",
        "    output_data = interpreter.get_tensor(output_details[0]['index'])\n",
        "    predictions.append((output_data[0]))\n",
        "  et = time.time()\n",
        "\n",
        "  predictions = np.array(predictions)\n",
        "  predictions = predictions * test_med + test_min\n",
        "\n",
        "  print(PATH,\"Details : \")  \n",
        "  print(\"Time taken to predict : \",et-st)\n",
        "  print (\"MAE:\", MAE(l_real,(np.squeeze(np.array(predictions)))) , \"Before : \",MAE(p_real, l_real))\n",
        "  print (\"MAPE:\", MAPE(l_real,(np.squeeze(np.array(predictions)))),\"Before : \",MAPE(p_real, l_real))\n",
        "  print (\"RMSE:\", RMSE(l_real,(np.squeeze(np.array(predictions)))),\"Before : \",RMSE(p_real, l_real))\n",
        "  \n",
        "  process = psutil.Process(os.getpid())\n",
        "  print(os.getpid())\n",
        "  print('Program takes',process.memory_info().rss / 1024 ** 2, 'mega-bytes of RAM')\n",
        "  print()\n",
        "  print()\n",
        "\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "TmfX3Xq_7l6C"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import tensorflow as tf\n",
        "import time\n",
        "def interpret_tflite_full_load(PATH):\n",
        "  # Load the TFLite model and allocate tensors.\n",
        "  interpreter = tf.lite.Interpreter(model_path=PATH)\n",
        "  interpreter.allocate_tensors()\n",
        "\n",
        "  # Get input and output tensors.\n",
        "  input_details = interpreter.get_input_details()\n",
        "  output_details = interpreter.get_output_details()\n",
        "\n",
        "  interpreter.resize_tensor_input(0,[2016,1,1], strict=True)\n",
        "  interpreter.resize_tensor_input(1,[2016,1,4,1], strict=True)\n",
        "  interpreter.resize_tensor_input(2,[2016,1,1], strict=True)\n",
        "\n",
        "\n",
        "  print((input_details))\n",
        "\n",
        "  # Test the model on random input data.\n",
        "  predictions = []\n",
        "  \n",
        "  st = time.time()\n",
        "  \n",
        "  interpreter.allocate_tensors()\n",
        "  interpreter.set_tensor(0,test_d.astype(np.float32))\n",
        "  interpreter.set_tensor(1,test_data.astype(np.float32) )\n",
        "  interpreter.set_tensor(2,test_w.astype(np.float32))\n",
        "\n",
        "  interpreter.invoke()\n",
        "  x1=interpreter.get_tensor(output_details[0]['index'])\n",
        "\n",
        "  # for i in range(2016):\n",
        "  #   interpreter.set_tensor(input_details[0]['index'], np.expand_dims(test_d[i].astype(np.float32),axis=0))\n",
        "  #   interpreter.set_tensor(input_details[1]['index'], np.expand_dims(test_data[i].astype(np.float32),axis=0))\n",
        "  #   interpreter.set_tensor(input_details[2]['index'], np.expand_dims(test_w[i].astype(np.float32),axis=0))\n",
        "  #   interpreter.invoke()\n",
        "  #   # The function `get_tensor()` returns a copy of the tensor data.\n",
        "  #   # Use `tensor()` in order to get a pointer to the tensor.\n",
        "  #   output_data = interpreter.get_tensor(output_details[0]['index'])\n",
        "  #   predictions.append((output_data[0]))\n",
        "  print(x1.shape)\n",
        "  predictions = (np.expand_dims(np.squeeze(x1),axis=1))\n",
        "  et = time.time()\n",
        "\n",
        "  predictions = np.array(predictions)\n",
        "  predictions = predictions * test_med + test_min\n",
        "\n",
        "  print(PATH,\"Details : \")  \n",
        "  print(\"Time taken to predict : \",et-st)\n",
        "  print (\"MAE:\", MAE(l_real,(np.squeeze(np.array(predictions)))) , \"Before : \",MAE(p_real, l_real))\n",
        "  print (\"MAPE:\", MAPE(l_real,(np.squeeze(np.array(predictions)))),\"Before : \",MAPE(p_real, l_real))\n",
        "  print (\"RMSE:\", RMSE(l_real,(np.squeeze(np.array(predictions)))),\"Before : \",RMSE(p_real, l_real))\n",
        "  \n",
        "  process = psutil.Process(os.getpid())\n",
        "  print(os.getpid())\n",
        "  print('Program takes',process.memory_info().rss / 1024 ** 2, 'mega-bytes of RAM')\n",
        "  print()\n",
        "  print()\n",
        "\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "G0duvkGil7G3",
        "outputId": "2c06ae03-7b40-41f8-e887-34b321b5affd"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "[{'name': 'serving_default_auxiliary_input_d:0', 'index': 0, 'shape': array([1, 1, 1], dtype=int32), 'shape_signature': array([-1,  1,  1], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}, {'name': 'serving_default_main_input:0', 'index': 1, 'shape': array([1, 1, 4, 1], dtype=int32), 'shape_signature': array([-1,  1,  4,  1], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}, {'name': 'serving_default_auxiliary_input_w:0', 'index': 2, 'shape': array([1, 1, 1], dtype=int32), 'shape_signature': array([-1,  1,  1], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]\n",
            "./conv_lstm_model_QA.tflite Details : \n",
            "Time taken to predict :  0.8500440120697021\n",
            "MAE: 4.429304726540097 Before :  4.9151650709765295\n",
            "MAPE: 0.08190187044202003 Before :  0.09779064263407072\n",
            "RMSE: 6.215104289468763 Before :  6.950996520508941\n",
            "72\n",
            "Program takes 2557.83984375 mega-bytes of RAM\n",
            "\n",
            "\n",
            "[{'name': 'serving_default_auxiliary_input_d:0', 'index': 0, 'shape': array([1, 1, 1], dtype=int32), 'shape_signature': array([-1,  1,  1], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}, {'name': 'serving_default_main_input:0', 'index': 1, 'shape': array([1, 1, 4, 1], dtype=int32), 'shape_signature': array([-1,  1,  4,  1], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}, {'name': 'serving_default_auxiliary_input_w:0', 'index': 2, 'shape': array([1, 1, 1], dtype=int32), 'shape_signature': array([-1,  1,  1], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]\n",
            "(2016, 1)\n",
            "./conv_lstm_model_QA.tflite Details : \n",
            "Time taken to predict :  0.03670859336853027\n",
            "MAE: 4.429304766749579 Before :  4.9151650709765295\n",
            "MAPE: 0.08190187813156809 Before :  0.09779064263407072\n",
            "RMSE: 6.215104248981528 Before :  6.950996520508941\n",
            "72\n",
            "Program takes 2560.6484375 mega-bytes of RAM\n",
            "\n",
            "\n"
          ]
        }
      ],
      "source": [
        "interpret_tflite('./conv_lstm_model_QA.tflite')\n",
        "interpret_tflite_full_load('./conv_lstm_model_QA.tflite')"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}