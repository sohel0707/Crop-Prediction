import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
# marketData = pd.read_csv('MarketPrice/Apple_2020.csv')
# print(marketData.sort_values(['arrival_date'])['arrival_date'])
import pickle
# marketSavedModel = "marketSavedModel.pkl"
best =0
uniqueAttributesInColumns =[]
# def getAccurancyOfMarketPrice(crop_name):
#     enc = LabelEncoder()
#
#     isCropPresent = False
#     files = os.listdir('MarketPrice')
#     for file in files:
#         name, ext = file.split('.')
#         if name == crop_name:
#             isCropPresent = True
#             break
#     if not isCropPresent:
#         return 0
#
#     marketData = pd.read_csv('MarketPrice/' + crop_name + '.csv')
#     marketData['arrival_date'] = pd.to_datetime(marketData['arrival_date'])
#     marketData.sort_values(['arrival_date'], inplace=True)
#
#     marketData['state'] = enc.fit_transform(marketData['state'])
#     marketData['district'] = enc.fit_transform(marketData['district'])
#     marketData['market'] = enc.fit_transform(marketData['market'])
#     marketData['commodity'] = enc.fit_transform(marketData['commodity'])
#     marketData['variety'] = enc.fit_transform(marketData['variety'])
#
#     y = marketData['modal_price']
#     X = marketData.drop(['arrival_date', 'min_price', 'max_price', 'modal_price'], axis=1)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
#
#     model = DecisionTreeRegressor()
#     model.fit(X, y)
#     y_pred = model.predict(X_test)
#     # print(list(zip(y_pred,y_test)))
#     return model.score(X_test, y_test) * 100
def saveModels(crop_name):
    global best,uniqueAttributesInColumns
    enc = LabelEncoder()

    isCropPresent = False
    is2020,is2021=False,False
    files = os.listdir('MarketPrice')
    for file in files:
        name,ext = file.split('.')
        if name==crop_name+'2020':
            isCropPresent = True
            is2020=True
        if name==crop_name+'2021':
            isCropPresent = True
            is2021=True

    if not isCropPresent:
        return 0
    if is2020 and is2021:
        marketData1 = pd.read_csv('MarketPrice/' + crop_name + '2020.csv')
        marketData2 = pd.read_csv('MarketPrice/' + crop_name + '2021.csv')
        marketData = pd.concat([marketData1,marketData2],ignore_index=True)
    elif is2020:
        marketData = pd.read_csv('MarketPrice/' + crop_name + '2020.csv')
    else:
        marketData = pd.read_csv('MarketPrice/' + crop_name + '2021.csv')

    # marketData = pd.read_csv('MarketPrice/' + crop_name + '.csv')

    marketData['arrival_date'] = pd.to_datetime(marketData['arrival_date'])
    marketData.sort_values(['arrival_date'],inplace=True)

    marketData['state']=enc.fit_transform(marketData['state'])
    np.save('./classes/state_classes.npy', enc.classes_,allow_pickle=True)

    marketData['district']=enc.fit_transform(marketData['district'])
    np.save('./classes/district_classes.npy', enc.classes_,allow_pickle=True)

    marketData['market']=enc.fit_transform(marketData['market'])
    np.save('./classes/market_classes.npy', enc.classes_,allow_pickle=True)

    marketData['commodity']=enc.fit_transform(marketData['commodity'])
    np.save('./classes/commodity_classes.npy', enc.classes_,allow_pickle=True)

    marketData['variety']=enc.fit_transform(marketData['variety'])
    np.save('./classes/variety_classes.npy', enc.classes_,allow_pickle=True)





    y = marketData['modal_price']
    X = marketData.drop(['arrival_date','min_price','max_price','modal_price'],axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # print(list(zip(y_pred,y_test)))
    cur = model.score(X_test,y_test)*100
    if cur>best:
        best = cur
        with open('./models/'+crop_name, 'wb') as file:
            pickle.dump(model, file)
    return cur
    # print("Decision Tree Regression Model accuracy(in %):", model.score(X_test,y_test)*100)

def getCurMarketPriceOfCrop(crop_name,data):
    if not os.path.exists('./models/'+crop_name): return 0
    enc = LabelEncoder()
    with open('./models/'+crop_name, 'rb') as f:
        model = pickle.load(f)
    enc.classes_= np.load('./classes/state_classes.npy',allow_pickle=True)
    stateAfterEncoding = enc.transform([data['state']])

    enc.classes_ = np.load('./classes/market_classes.npy',allow_pickle=True)
    print(enc.classes_)
    marketAfterEncoding = enc.transform([data['market']])


    enc.classes_ = np.load('./classes/commodity_classes.npy',allow_pickle=True)
    print(enc.classes_)
    commodityAfterEncoding = enc.transform([data['commodity']])


    enc.classes_ = np.load('./classes/variety_classes.npy',allow_pickle=True)
    print(enc.classes_)
    varietyAfterEncoding = enc.transform([data['variety']])


    enc.classes_ = np.load('./classes/district_classes.npy',allow_pickle=True)
    print(enc.classes_)
    districtAfterEncoding = enc.transform([data['district']])


    # stateAfterEncoding = uniqueAttributesInColumns[0].find(data['state'])
    # districtAfterEncoding = uniqueAttributesInColumns[1].find(data['district'])
    # marketAfterEncoding = uniqueAttributesInColumns[2].find(data['market'])
    # commodityAfterEncoding = uniqueAttributesInColumns[3].find(data['commodity'])
    # varietyAfterEncoding = uniqueAttributesInColumns[4].find(data['variety'])
    data = [[stateAfterEncoding[0],districtAfterEncoding[0],marketAfterEncoding[0],commodityAfterEncoding[0],varietyAfterEncoding[0]]]
    return model.predict(data)


# for  i in os.listdir('MarketPrice'):
#     globals()['best']=0
#     name, ext = i.split('.')
#     print(name[:-4])
#     for j in range(50):
#         saveModels(name[:-4])
#     print(globals()['best'])
# print(uniqueAttributesInColumns)
# print(getCurMarketPriceOfCrop('apple',{'state':'Punjab','district':'Ferozpur','market':"Firozepur City",'commodity':'Apple','variety':'Other'}))
