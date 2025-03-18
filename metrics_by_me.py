from keras import backend as K
import tensorflow as tf


def recall_m(y_true, y_pred):
    m = tf.keras.metrics.Recall()
    m.update_state(y_true=y_true, y_pred=y_pred)
    return m.result().numpy()


def precision_m(y_true, y_pred):
    m = tf.keras.metrics.Precision(thresholds=0.5)
    m.update_state(y_true=y_true, y_pred=y_pred)
    return m.result().numpy()


def f1_m(recall, precision):
    return 2 * (precision * recall) / (precision + recall)