import pandas as pd
import os
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

best =0
uniqueAttributesInColumns =[]


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

    # marketData['arrival_date'] = pd.to_datetime(marketData['arrival_date'])
    # marketData.sort_values(['arrival_date'],inplace=True)

    marketData['state']=enc.fit_transform(marketData['state'])
    np.save('./classes/'+crop_name+'state_classes.npy', enc.classes_,allow_pickle=True)
    marketData['district']=enc.fit_transform(marketData['district'])
    np.save('./classes/'+crop_name+'district_classes.npy', enc.classes_,allow_pickle=True)
    marketData['market']=enc.fit_transform(marketData['market'])
    np.save('./classes/'+crop_name+'market_classes.npy', enc.classes_,allow_pickle=True)
    # marketData['commodity']=enc.fit_transform(marketData['commodity'])
    # np.save('./classes/'+crop_name+'commodity_classes.npy', enc.classes_,allow_pickle=True)
    #
    # marketData['variety']=enc.fit_transform(marketData['variety'])
    # np.save('./classes/'+crop_name+'variety_classes.npy', enc.classes_,allow_pickle=True)
    y = marketData['modal_price']
    X = marketData.drop(['arrival_date','min_price','max_price','modal_price','commodity','variety'],axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
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
    enc.classes_= np.load('./classes/'+crop_name+'state_classes.npy',allow_pickle=True)
    # print(enc.classes_)
    if data['state'] not in enc.classes_:
        return 0
    stateAfterEncoding = enc.transform([data['state']])

    enc.classes_ = np.load('./classes/'+crop_name+'market_classes.npy',allow_pickle=True)
    # print(enc.classes_)
    if data['market'] not in enc.classes_:
        return 0
    marketAfterEncoding = enc.transform([data['market']])


    enc.classes_ = np.load('./classes/'+crop_name+'district_classes.npy',allow_pickle=True)
    # print(enc.classes_)
    if data['district'] not in enc.classes_:
        return 0
    districtAfterEncoding = enc.transform([data['district']])

    data = [[stateAfterEncoding[0],districtAfterEncoding[0],marketAfterEncoding[0]]]
    return model.predict(data)


def main():
    typesOfCrops = set()
    for i in os.listdir('MarketPrice'):
        name, ext = i.split('.')
        nameOfCrop = name[:-4]
        typesOfCrops.add(nameOfCrop)
    lenOfDifferentTypesOfCrop = len(typesOfCrops)

    for ind,nameOfCrop in enumerate(typesOfCrops):

        globals()['best'] = 0
        for j in range(50):
            saveModels(nameOfCrop)
        print("--"*(ind+1)+"{} %".format(int(((ind+1)*100)/lenOfDifferentTypesOfCrop)), flush=True,end= "\r" if ind!=(lenOfDifferentTypesOfCrop-1) else "\n")
        # print(globals()['best'])


if __name__=='__main__':
    main()
