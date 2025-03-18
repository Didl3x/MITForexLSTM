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

data = 'data/'
X_train, y_train, X_cv, y_cv, X_test, y_test, init_bias, class_weight = data_formatting.transform(data=data)
output_bias = tf.keras.initializers.Constant(init_bias)
model1 = Sequential([
    InputLayer(batch_input_shape=(24, 300, 16)),
    LSTM(units=64, stateful=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    Dense(units=8, activation='tanh', kernel_regularizer=tf.keras.regularizers.L2(0.16)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    Dense(units=1, activation='sigmoid', bias_initializer=output_bias)
])

model1.summary()

cp = ModelCheckpoint('ModelV6/', save_best_only=True)

# model1 = tf.keras.models.load_model('ModelV1/')
model1.compile(loss=BinaryCrossentropy(), optimizer=AdamW(learning_rate=0.0001), metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

model1.fit(X_train, y_train, validation_data=(X_cv, y_cv), epochs=150, batch_size=24, callbacks=[cp], class_weight=class_weight)
