from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import AdamW
import tensorflow as tf
import data_formatting
import pandas as pd
from sklearn.preprocessing import Normalizer

data = pd.read_csv('data.csv')
X_train, y_train, X_cv, y_cv, X_test, y_test, init_bias = data_formatting.transform(data=data)
output_bias = tf.keras.initializers.Constant(init_bias)
model1 = Sequential([
    InputLayer(batch_input_shape=(6, 90, 15)),
    LSTM(units=64, return_sequences=True, stateful=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    LSTM(units=64, return_sequences=True, stateful=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    LSTM(units=64, stateful=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    Dense(units=32, activation='tanh'),
    tf.keras.layers.BatchNormalization(),
    Dense(units=1, activation='sigmoid', bias_initializer=output_bias)
])

model1.summary()

cp = ModelCheckpoint('ModelOverfitAttempt/', save_best_only=True)

#model1 = tf.keras.models.load_model('model2/')
model1.compile(loss=BinaryCrossentropy(), optimizer=AdamW(learning_rate=0.000001), metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

model1.fit(X_train, y_train, validation_data=(X_cv, y_cv), epochs=1000, batch_size=6, callbacks=[cp])

