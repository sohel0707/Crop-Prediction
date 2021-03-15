from sklearn.ensemble import RandomForestClassifier
import pickle
def calculate_accurancy(data):
    y = data['label']
    X = data.drop(['label'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    from sklearn.naive_bayes import GaussianNB

    # model = GaussianNB()
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    # for i in model.predict_proba(X_test):
    #     print(normalize(i))
    # print(normalize(model.predict_proba(X_test)))
    y_pred = model.predict(X_test)
    # print(list(zip(y_test,y_pred)))
    from sklearn import metrics
    print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred) * 100)


# def normalize(probs):
#     prob_factor = 1 / sum(probs)
#     return [prob_factor * p for p in probs]

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
#
crop_data = pd.read_csv("./crop_prediction_dataset.csv")
# # calculate_accurancy(crop_data)
#
# # print(market_data.columns)
# # print(np.unique(y))
# # normalized_arr = preprocessing.normalize([np.array(crop_data['humidity'])])
# # plt.scatter(crop_data['rainfall'],crop_data['label'])
# # plt.scatter(normalized_arr,crop_data['label'])
# # plt.show()
#
# y = crop_data['label']
# X = crop_data.drop(['label'], axis=1)
# # from sklearn.naive_bayes import GaussianNB
# #
# model = RandomForestClassifier()
# model.fit(X, y)
# with open('./models/' + 'crop', 'wb') as file:
    # pickle.dump(model, file)
# #
import random
# # model.predict_proba()
# l = model.predict_proba([[random.random()*100,random.random()*100,random.random()*100,random.random()*100]])
# print(model.classes_)
# print(l)
# l = [[0.00000000e+000, 0.00000000e+000 ,0.00000000e+000 ,0.00000000e+000,
#   0.00000000e+000, 0.00000000e+000 ,0.00000000e+000, 0.00000000e+000,
#   0.00000000e+000, 0.00000000e+000 ,1.00000000e+000 ,0.00000000e+000,
#   0.00000000e+000, 0.00000000e+000 ,0.00000000e+000 ,3.65587709e-276,
#   0.00000000e+000, 0.00000000e+000 ,0.00000000e+000, 0.00000000e+000,
#   0.00000000e+000, 0.00000000e+000 ,0.00000000e+000 ,0.00000000e+000,
#   0.00000000e+000, 0.00000000e+000 ,0.00000000e+000, 0.00000000e+000,
#   0.00000000e+000, 0.00000000e+000 ,0.00000000e+000]]


# for i in l[0]:
#     # print(type(i))
#     print('{:.3f}'.format(i),end=", ")



