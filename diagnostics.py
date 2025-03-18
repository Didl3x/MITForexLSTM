import tensorflow as tf
from tensorflow.keras.models import load_model
import data_formatting
import pandas as pd
import matplotlib.pyplot as plt
import random
import metrics_by_me
import seaborn as sn
from sklearn import metrics
import numpy as np

data = 'data/'
X_train, y_train, X_cv, y_cv, X_test, y_test, init_bias, class_weight = data_formatting.transform(data=data)

model = load_model('ModelV6/')

# x_da = pd.DataFrame(X_train)
# x_da.to_csv('x_train.csv')

# loss, acc, pre, rec = model.evaluate(X_train, y_train, verbose=2, batch_size=32)
train_predictions = model.predict(X_train, batch_size=24).flatten()
cv_predictions = model.predict(X_cv, batch_size=24).flatten()
test_predictions = model.predict(X_test, batch_size=24).flatten()
train_results = pd.DataFrame(data={'Predictions': train_predictions, 'Actuals': y_train})
cv_results = pd.DataFrame(data={'Predictions': cv_predictions, 'Actuals': y_cv})
test_results = pd.DataFrame(data={'Predictions': test_predictions, 'Actuals': y_test})

cv_results.to_csv('cv.csv')
test_results.to_csv('test.csv')
correct = {
    "test_results": 0,
    "train_results": 0,
    "cv_results": 0
}
correct_base = 0
answers = {
    "test_results": 0,
    "train_results": 0,
    "cv_results": 0
}
predictions = [[test_results, 'test_results'], [train_results, 'train_results'], [cv_results, 'cv_results']]
for prediction in predictions:
    for i in range(len(prediction[0]['Predictions'])):
        if prediction[0]['Predictions'][i] > 0.5:
            answers[prediction[1]] = 1
        else:
            answers[prediction[1]] = 0

        if answers[prediction[1]] == prediction[0]['Actuals'][i]:
            correct[prediction[1]] += 1

for i in range(len(test_results)):
    if 1 == test_results['Actuals'][i]:
        correct_base += 1

print(f"""
Baseline Accuracy 1: {correct_base/len(test_results['Predictions'])}, 0: {1 - correct_base/len(test_results['Predictions'])}

Train Accuracy: {correct['train_results']/len(train_results['Predictions'])}
Train Precision: {metrics_by_me.precision_m(y_train, train_predictions)}
Train Recall: {metrics_by_me.recall_m(y_train, train_predictions)}
Train F1 Score: {metrics_by_me.f1_m(metrics_by_me.recall_m(y_train, train_predictions), metrics_by_me.precision_m(y_train, train_predictions))}

CV Accuracy: {correct['cv_results']/len(cv_results['Predictions'])}
CV Precision: {metrics_by_me.precision_m(y_cv, cv_predictions)}
CV Recall: {metrics_by_me.recall_m(y_cv, cv_predictions)}
CV F1 Score: {metrics_by_me.f1_m(metrics_by_me.recall_m(y_cv, cv_predictions), metrics_by_me.precision_m(y_cv, cv_predictions))}

Test Accuracy: {correct['test_results']/len(test_results['Predictions'])}
Test Precision: {metrics_by_me.precision_m(y_test, test_predictions)}
Test Recall: {metrics_by_me.recall_m(y_test, test_predictions)}
Test F1 Score: {metrics_by_me.f1_m(metrics_by_me.recall_m(y_test, test_predictions), metrics_by_me.precision_m(y_test, test_predictions))}
""")

binary_train_predictions = np.ndarray((240000,))
for i in range(len(train_predictions)):
    if train_predictions[i] > 0.5:
        binary_train_predictions[i] = 1
    else:
        binary_train_predictions[i] = 0

cm = metrics.confusion_matrix(y_train, binary_train_predictions)
print(cm)
df_cm = pd.DataFrame(cm, range(2), range(2))
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

binary_test_predictions = np.ndarray((47976,))
for i in range(len(test_predictions)):
    if test_predictions[i] > 0.5:
        binary_test_predictions[i] = 1
    else:
        binary_test_predictions[i] = 0
cm = metrics.confusion_matrix(y_test, binary_test_predictions)
print(cm)
# df_cm2 = pd.DataFrame(cm, range(2), range(2))
# plt.figure(figsize=(10,7))
# sn.set(font_scale=1.4) # for label size
# sn.heatmap(df_cm2, annot=True, annot_kws={"size": 16}) # font size

plt.show()

# fig, ax = plt.subplots(5, 2)
# ax[0][0].plot(data['Open'])
# ax[1][0].plot(data['Close'])
# ax[2][0].plot(data['High'])
# ax[3][0].plot(data['Low'])
# ax[4][0].plot(data['Volume'])
# for o in range(5):
#     for i in X_train[o].T:
#         if i[0] < 3 and i[0] > 0.5:
#             ax[o][1].plot(i)
#
#
#
# plt.show()

# fig, ax = plt.subplots(3)
# ax[0].plot(train_results['Train Predictions'])
# ax[0].plot(train_results['Actuals'])
#
# ax[1].plot(cv_results['CV Predictions'])
# ax[1].plot(cv_results['Actuals'])
#
# ax[2].plot(test_results['Test Predictions'])
# ax[2].plot(test_results['Actuals'])
# plt.show()
