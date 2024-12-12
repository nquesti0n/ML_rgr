import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import rand_score

with open('C:/Users/Anastasia/Desktop/ML_RGR-main/models/OneHotEncoder.pkl', 'rb') as file: 
    ColumnTransform = pickle.load(file)

uploaded_file = st.file_uploader("Выберите файл датасета")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Загруженный датасет:", df)

else:
    st.write("Датасет Rain in Australia")
    df = pd.read_csv('C:/Users/Anastasia/Desktop/ML_RGR-main/data/weatherAUS.csv')

df.dropna(inplace=True,ignore_index=True)
f = lambda x : str(x)[5:7]
df['Date'] = df['Date'].transform(f)
df['Date'] = df['Date'].astype(int)
f = lambda x : 0 if (x == "No") else 1
df['RainToday'] = df['RainToday'].transform(f)
df['RainToday'] = df['RainToday'].astype(int)

df['RainTomorrow'] = df['RainTomorrow'].transform(f)
df['RainTomorrow'] = df['RainTomorrow'].astype(int)
encoded_features = ColumnTransform.transform(df)
data1=df.drop(['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'],axis=1)
dataclass = pd.concat([
    data1,
    encoded_features
], axis=1)
x_class=dataclass.drop(['RainTomorrow'],axis=1)
y_class=dataclass['RainTomorrow']



button_clicked_metrics = st.button("Расчитать точность моделей на датасете")

if button_clicked_metrics:
    with open('C:/Users/Anastasia/Desktop/ML_RGR-main/models/SVC.plk', 'rb') as file:
        svc_model = pickle.load(file)

    with open('C:/Users/Anastasia/Desktop/ML_RGR-main/models/Bagging.plk', 'rb') as file:
        bagging_model = pickle.load(file)

    with open('C:/Users/Anastasia/Desktop/ML_RGR-main/models/GradientBoosting.plk', 'rb') as file:
        gradient_model = pickle.load(file)

    with open('C:/Users/Anastasia/Desktop/ML_RGR-main/models/Stacking_model.plk', 'rb') as file:
        stacking_model = pickle.load(file)

    from tensorflow.keras.models import load_model
    nn_model = load_model('C:/Users/Anastasia/Desktop/ML_RGR-main/models/NN.h5')


    st.header("SVC:")
    svc_pred = svc_model.predict(x_class)
    st.write('Accuracy: ',f"{accuracy_score(y_class, svc_pred)}")


    st.header("bagging:")
    bagging_pred = bagging_model.predict(x_class)
    st.write('Accuracy: ',f"{accuracy_score(y_class, bagging_pred)}")

    st.header("gradient:")
    gradient_pred = gradient_model.predict(x_class)
    st.write('Accuracy: ',f"{accuracy_score(y_class, gradient_pred)}")

    st.header("Perceptron:")
    nn_pred = [np.argmax(pred) for pred in nn_model.predict(x_class, verbose=None)]
    st.write('Accuracy: ',f"{accuracy_score(y_class, nn_pred)}")

    st.header("Stacking:")
    stacking_pred = stacking_model.predict(x_class)
    st.write('Accuracy: ',f"{accuracy_score(y_class, stacking_pred)}")


st.title("Получить предсказание дождя.")

st.header("Date")
Date = st.number_input("Число:", value=2012, min_value=1900, max_value=2100)

st.header("Location")
locations = ['Cobar', 'CoffsHarbour', 'Moree', 'NorfolkIsland', 'Sydney',
       'SydneyAirport', 'WaggaWagga', 'Williamtown', 'Canberra', 'Sale',
       'MelbourneAirport', 'Melbourne', 'Mildura', 'Portland', 'Watsonia',
       'Brisbane', 'Cairns', 'Townsville', 'MountGambier', 'Nuriootpa',
       'Woomera', 'PerthAirport', 'Perth', 'Hobart', 'AliceSprings',
       'Darwin']
Location = st.selectbox("Город", locations)

st.header("MinTemp")
MinTemp = st.number_input("Число:", value=20.9)

st.header("MaxTemp")
MaxTemp = st.number_input("Число:", value=37.8)

st.header("Rainfall")
Rainfall = st.number_input("Число:", value=2)

st.header("Evaporation")
Evaporation = st.number_input("Число:", value=12.8)

st.header("Sunshine")
Sunshine = st.number_input("Число:", value=13.2)

st.header("WindGustDir")
dirs=['SSW', 'S', 'NNE', 'WNW', 'N', 'SE', 'ENE', 'NE', 'E', 'SW', 'W',
       'WSW', 'NNW', 'ESE', 'SSE', 'NW']
WindGustDir = st.selectbox("Направление", dirs)

st.header("WindGustSpeed")
WindGustSpeed = st.number_input("Число:", value=30)

st.header("WindDir9am")
dirs2=['ENE', 'SSE', 'NNE', 'WNW', 'NW', 'N', 'S', 'SE', 'NE', 'W', 'SSW',
       'E', 'NNW', 'ESE', 'WSW', 'SW']
WindDir9am = st.selectbox("Направление", dirs2)

st.header("WindDir3pm")
dirs3=['SW', 'SSE', 'NNW', 'WSW', 'WNW', 'S', 'ENE', 'N', 'SE', 'NNE',
       'NW', 'E', 'ESE', 'NE', 'SSW', 'W']
WindDir3pm = st.selectbox("Направление", dirs3)

st.header("WindSpeed9am")
WindSpeed9am = st.number_input("Число:", value=11)

st.header("WindSpeed3pm")
WindSpeed3pm = st.number_input("Число:", value=7)

st.header("Humidity9am")
Humidity9am = st.number_input("Число:", value=27)

st.header("Humidity3pm")
Humidity3pm = st.number_input("Число:", value=9)

st.header("Pressure9am")
Pressure9am = st.number_input("Число:", value=1012.6)

st.header("Pressure3pm")
Pressure3pm = st.number_input("Число:", value=1010.1)

st.header("Cloud9am")
Cloud9am = st.number_input("Число:", value=0.1)

st.header("Cloud3pm")
Cloud3pm = st.number_input("Число:", value=1)

st.header("Temp9am")
Temp9am = st.number_input("Число:", value=29.8)

st.header("Temp3pm")
Temp3pm = st.number_input("Число:", value=36.4)

st.header("RainToday")
RainToday = st.number_input("Число:", value=0, min_value=0, max_value=1)

data = pd.DataFrame({'Date': [Date],
                    'Location': [Location],
                    'MinTemp': [MinTemp],
                    'MaxTemp': [MaxTemp],
                    'Rainfall': [Rainfall],
                    'Evaporation': [Evaporation],
                    'Sunshine': [Sunshine],
                    'WindGustDir': [WindGustDir],
                    'WindGustSpeed': [WindGustSpeed],
                    'WindDir9am': [WindDir9am],
                    'WindDir3pm': [WindDir3pm],
                    'WindSpeed9am': [WindSpeed9am],
                    'WindSpeed3pm': [WindSpeed3pm],
                    'Humidity9am': [Humidity9am],    
                    'Humidity3pm': [Humidity3pm],   
                    'Pressure9am': [Pressure9am],   
                    'Pressure3pm': [Pressure3pm],   
                    'Cloud9am': [Cloud9am],   
                    'Cloud3pm': [Cloud3pm],   
                    'Temp9am': [Temp9am],   
                    'Temp3pm': [Temp3pm],   
                    'RainToday': [RainToday],  
                    'RainTomorrow': [0]       
                    })


encoded_features = ColumnTransform.transform(data)
data1=data.drop(['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'],axis=1)
df = pd.concat([
    data1,
    encoded_features
], axis=1)
df=df.drop(['RainTomorrow'],axis=1)




button_clicked = st.button("Предсказать")

if button_clicked:
    with open('C:/Users/Anastasia/Desktop/ML_RGR-main/models/SVC.plk', 'rb') as file:
        svc_model = pickle.load(file)

    with open('C:/Users/Anastasia/Desktop/ML_RGR-main/models/Bagging.plk', 'rb') as file:
        bagging_model = pickle.load(file)

    with open('C:/Users/Anastasia/Desktop/ML_RGR-main/models/GradientBoosting.plk', 'rb') as file:
        gradient_model = pickle.load(file)

    with open('C:/Users/Anastasia/Desktop/ML_RGR-main/models/Stacking_model.plk', 'rb') as file:
        stacking_model = pickle.load(file)

    from tensorflow.keras.models import load_model
    nn_model = load_model('C:/Users/Anastasia/Desktop/ML_RGR-main/models/NN.h5')


    st.header("SVC:")
    pred =[]
    svc_pred = svc_model.predict(df)[0]
    pred.append(int(svc_pred))
    st.write(f"{svc_pred}")


    st.header("bagging:")
    bagging_pred = bagging_model.predict(df)[0]
    pred.append(int(bagging_pred))
    st.write(f"{bagging_pred}")

    st.header("gradient:")
    gradient_pred = gradient_model.predict(df)[0]
    pred.append(int(gradient_pred))
    st.write(f"{gradient_pred}")

    st.header("Perceptron:")
    nn_pred = round(nn_model.predict(df)[0][0])
    pred.append(nn_pred)
    st.write(f"{nn_pred}")

    st.header("Stacking:")
    stacking_pred = stacking_model.predict(df)[0]
    pred.append(int(stacking_pred))
    st.write(f"{stacking_model.predict(df)[0]}")
