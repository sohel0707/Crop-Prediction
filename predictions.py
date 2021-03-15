import market_pred
import pickle
import streamlit as st
import numpy as np
# from sklearn.ensemble import RandomForestClassifier

def cmp(prob):
    return prob>=0.2
with open('./models/' + 'crop', 'rb') as f:
    model = pickle.load(f)

def predict_crop(state,district,market,commodity,variety,temperature,humidity,ph,rainfall):
    # temperature,humidity,ph,rainfall = 0,0,0,0
    # state,district,market,commodity,variety = "","","","",""
    # print(temperature,humidity,ph,rainfall)
    temperature,humidity,ph,rainfall = float(temperature),float(humidity),float(ph),float(rainfall)
    probabilities = model.predict_proba([[temperature,humidity,ph,rainfall]])[0]
    classes = model.classes_
    considered_classes = []
    for ind,probability in enumerate(probabilities):
        if cmp(probability):
            considered_classes.append(classes[ind])
    maxRevenueGenratedClass = ""
    maxRevenue = 0
    for class_ in considered_classes:
        curRevenue = market_pred.getCurMarketPriceOfCrop(class_,{'state':state,'district':district,'market':market,'commodity':commodity,'variety':variety})
        if maxRevenue<curRevenue:
            maxRevenue = curRevenue
            maxRevenueGenratedClass = class_
    return "maximum revenue generated crop is"+maxRevenueGenratedClass if maxRevenueGenratedClass!="" else "Sorry We Cant Find Maximum revenue generated crop"

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
    commodity=st.text_input("commodity","type here")
    variety=st.text_input("variety","type here")
    temperature=st.text_input("temperature","type here")
    humidity=st.text_input("humidity","type here")
    ph=st.text_input("ph","type here")
    rainfall=st.text_input("rainfall","type here")
    if st.button('predict'):
        crop = predict_crop(state,district,market,commodity,variety,temperature,humidity,ph,rainfall)
        st.success(crop)

if __name__=='__main__':
    main()



