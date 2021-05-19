import crop_pred
import market_pred
import pickle
import streamlit as st
import pandas as pd
import numpy as np
# from sklearn.ensemble import RandomForestClassifier

def minimumProbabilty(prob):
    return prob>=0.0001
with open('./models/' + 'crop', 'rb') as f:
    model = pickle.load(f)

def predict_crop(state,district,market,temperature,humidity,ph,rainfall):
    temperature,humidity,ph,rainfall = float(temperature),float(humidity),float(ph),float(rainfall)
    considered_classes = crop_pred.getCropsHavingMinimumProbability([temperature,humidity,ph,rainfall],minimumProbabilty)
    # probabilities = model.predict_proba([[temperature,humidity,ph,rainfall]])[0]
    # classes = model.classes_
    # considered_classes = []
    # # print(list(zip(probabilities,classes)))
    # for ind,probability in enumerate(probabilities):
    #     if cmp(probability):
    #         considered_classes.append(classes[ind])
    maxRevenueGenratedClass = ""
    maxRevenue = 0
    revenues = []
    for class_ in considered_classes:
        curRevenue = market_pred.getCurMarketPriceOfCrop(class_,{'state':state,'district':district,'market':market})
        # print(class_,curRevenue)
        if not (type(curRevenue)==int or type(curRevenue)==float ):
            curRevenue = curRevenue.tolist()
        if(type(curRevenue)==list):
            curRevenue =curRevenue[0]
        print(type(curRevenue))
        revenues.append(curRevenue)
        if maxRevenue<curRevenue:
            maxRevenue = curRevenue
            maxRevenueGenratedClass = class_
    chart_data = pd.DataFrame(revenues,considered_classes)

    return "maximum revenue generated crop is "+maxRevenueGenratedClass if maxRevenueGenratedClass!="" else "Sorry We Cant Find Maximum revenue generated crop",chart_data

def main():
    st.title("Crop Predition")
    html_temp = """
        <div style="background-color:gray;padding:10px">
        <h2 style="color:white;text-align:center;">Crop Prediction</h2>
        </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    state=st.text_input("State","type here")
    district=st.text_input("district","type here")
    market=st.text_input("market","type here")
    temperature=st.text_input("temperature","type here")
    humidity=st.text_input("humidity","type here")
    ph=st.text_input("ph","type here")
    rainfall=st.text_input("rainfall","type here")
    if st.button('predict'):
        crop,chart_data = predict_crop(state,district,market,temperature,humidity,ph,rainfall)
        st.success(crop)
        st.bar_chart(chart_data)

if __name__=='__main__':
    main()



