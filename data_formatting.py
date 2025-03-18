import numpy as np
import pandas as pd
import talib
from talib.abstract import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
import os


def transform(data):
    dfs = []  # Create a list to store DataFrames
    for datafile in os.listdir(f'{data}'):
        f = os.path.join(data, datafile)
        if os.path.isfile(f):
            new_df = pd.read_csv(f)
            new_df['Gmt time'] = pd.to_datetime(new_df['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f')  # Specify the format
            new_df.set_index('Gmt time', inplace=True)
            new_df = new_df.resample('S').asfreq()  # Resample to seconds
            new_df = new_df.interpolate(method='ffill')  # Forward fill missing data
            dfs.append(new_df)  # Append each DataFrame to the list
            print(new_df)
        print('OOO')

    # Concatenate all DataFrames in the list
    if not dfs:  # Check if the list is empty
        return None  # Return None if there are no DataFrames to concatenate
    df = pd.concat(dfs)

    print(df)  # Print the final concatenated DataFrame after the loop


    inputs = {
        'open': df['Open'].to_numpy(),
        'low': df['Low'].to_numpy(),
        'close': df['Close'].to_numpy(),
        'volume': df['Volume'].to_numpy(),
        'high': df['High'].to_numpy(),
    }

    sma = SMA(inputs, timeperiod=14)
    macd, macdsignal, macdhist = MACD(inputs, fasperiod=12, slowperiod=26, signalperiod=9)
    upperband, middleband, lowerband = BBANDS(inputs, timeperiod=5, nbdevup=2.0, nbdevdn=2.0, matype=0)
    rsi = RSI(inputs, timeperiod=14)
    slowk, slowd = STOCH(inputs['high'], inputs['low'], inputs['close'], fastk_period=5, fastd_period=3, fastd_matype=0)
    cci = CCI(inputs['high'], inputs['low'], inputs['close'], timeperiod=14)
    adx = ADX(inputs['high'], inputs['low'], inputs['close'], timeperiod=14)
    atr = ATR(inputs['high'], inputs['low'], inputs['close'], timeperiod=14)

    price_changes = np.ndarray(len(inputs['close']))
    print(price_changes.shape)
    price_last = inputs['close'][0]
    for i in range(len(inputs['close']) - 1):
        price_changes[i] = price_last - inputs['close'][i+1]
        price_last = inputs['close'][i+1]

    def df_to_X_y(df, window_size=300):
        df_as_np = df.to_numpy()
        X = []
        y = []
        for i in range(len(df_as_np)-window_size-59):
            row = [
                a for a in df_as_np[i:i+window_size]
            ]
            X.append(row)
            next_candle = df_as_np[i+window_size+59][0]
            label = int(next_candle > inputs['open'][i+window_size+50])
            y.append(label)
        print(np.array(y))
        print(np.array(X))
        return np.array(X), np.array(y)


    indicator_df = pd.DataFrame(data={
        #
        # 'open': inputs['open'][50:],
        # 'low': inputs['low'][50:],
        'close': inputs['close'][50:],
        'volume': inputs['volume'][50:],
        'pricechange': price_changes[50:],
        # 'high': inputs['high'][50:],
        'sma': sma[50:],
        'macd': macd[50:],
        'macdsignal': macdsignal[50:],
        'macdhist': macdhist[50:],
        'upperband': upperband[50:],
        'middleband': middleband[50:],
        'lowerband': lowerband[50:],
        'rsi': rsi[50:],
        'slowk': slowk[50:],
        'slowd': slowd[50:],
        'cci': cci[50:],
        'adx': adx[50:],
        'atr': atr[50:]
    })
    indicator_df.to_excel('Features.xlsx')
    X, y = df_to_X_y(df=indicator_df)
    print(X.shape, y.shape)

    # X_train, y_train = X[:800000], y[:800000]
    # X_cv, y_cv = X[800032:860000], y[800032:860000]
    # X_test, y_test = X[860049:], y[860049:]
    X_train, y_train = X[:240000], y[:240000]
    X_cv, y_cv = X[240000:280008], y[240000:280008]
    X_test, y_test = X[280024:328000], y[280024:328000]
    print(X_train.shape, y_train.shape, X_cv.shape, y_cv.shape, X_test.shape, y_test.shape)

    # subplots = 3
    # begin = 0
    # end = 100
    # fig, ax = plt.subplots(subplots)
    # ax[1].plot(macd[begin:end], color='g', label='MACD')
    # ax[1].plot(macdsignal[begin:end], color='r')
    # ax[1].plot(macdhist[begin:end], color='g')
    #
    # ax[2].plot(rsi[begin:end], color='r', label='RSI')
    # ax[0].plot(sma[begin:end], color='r', label='SMA')
    # ax[0].plot(inputs["close"][begin:end], color='cyan', label='price')
    # for band in [upperband, middleband, lowerband]:
    #     ax[0].plot(band[begin:end], color='b')
    #
    # plt.xlabel('Time')
    # plt.ylabel('Price')
    #
    # plt.legend()
    # plt.grid()
    #
    # plt.show()

    # Reshape your data to 2D (number of samples, number of features)
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    X_cv_2d = X_cv.reshape(-1, X_cv.shape[-1])
    X_test_2d = X_test.reshape(-1, X_test.shape[-1])

    # Create a scaler and fit it to the training data
    scaler = Normalizer()
    X_train_scaled_2d = scaler.fit_transform(X_train_2d)
    X_cv_scaled_2d = scaler.transform(X_cv_2d)
    X_test_scaled_2d = scaler.transform(X_test_2d)

    # Reshape the scaled data back to 3D
    X_train_scaled = X_train_scaled_2d.reshape(X_train.shape)
    X_cv_scaled = X_cv_scaled_2d.reshape(X_cv.shape)
    X_test_scaled = X_test_scaled_2d.reshape(X_test.shape)

    # zero_total = 0
    # one_total = 0
    # for i in y_train:
    #     if i == 1:
    #         one_total += 1
    #     else:
    #         zero_total += 1
    #
    # zero_total = 0
    # X_train_new = np.ndarray((one_total * 2, 300, 15))
    # y_train_new = np.ndarray((one_total * 2,))
    # i_change = 0
    # for i in range(len(X_train_scaled)):
    #     if y_train[i] == 0:
    #         if zero_total < one_total:
    #             X_train_new[i-i_change] = X_train_scaled[i]
    #             y_train_new[i-i_change] = y_train[i]
    #             zero_total += 1
    #         else:
    #             i_change += 1
    #     else:
    #         X_train_new[i-i_change] = X_train_scaled[i]
    #         y_train_new[i-i_change] = y_train[i]
    # print(zero_total)
    # X_train_scaled = X_train_new[:]
    # y_train = y_train_new[:]

    zero_total = 0
    one_total = 0
    for i in y_train:
        if i == 1:
            one_total += 1
        else:
            zero_total += 1

    class_weight = {
        0: one_total / (zero_total + one_total),
        1: zero_total / (zero_total + one_total)
    }


    print(f'Zero Total: {zero_total}, One Total: {one_total}')
    init_bias = np.log(one_total / zero_total)
    print(init_bias)
    print(y_train)
    return X_train_scaled, y_train, X_cv_scaled, y_cv, X_test_scaled, y_test, init_bias, class_weight


