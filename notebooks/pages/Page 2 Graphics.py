import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users/Anastasia/Desktop/ML_RGR-main/data/Weather_data.csv')

st.title("Датасет Rain in Australia")

st.header("Тепловая карта с корреляцией между основными признаками")

plt.figure(figsize=(12, 8))
selected_cols = ['RainTomorrow', 'RainToday', 'WindGustSpeed','Humidity9am','Temp9am', "Rainfall"]
selected_df = df[selected_cols]
sns.heatmap(selected_df.corr(), annot=True, cmap='coolwarm')
plt.title('Тепловая карта с корреляцией')
st.pyplot(plt)

st.header("Гистограммы")

columns = ['WindGustSpeed','Humidity9am','Rainfall']

for col in columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df.sample(5000)[col], bins=100, kde=True)
    plt.title(f'Гистограмма для {col}')
    st.pyplot(plt)


st.header("Ящики с усами ")
outlier = df[columns]
Q1 = outlier.quantile(0.25)
Q3 = outlier.quantile(0.75)
IQR = Q3-Q1
data_filtered = outlier[~((outlier < (Q1 - 1.5 * IQR)) |(outlier > (Q3 + 1.5 * IQR))).any(axis=1)]


for col in columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data_filtered[col])
    plt.title(f'{col}')
    plt.xlabel('Значение')
    st.pyplot(plt)



st.header("Графики для целевого признака и наиболее коррелирующих с ним некоторых признаков")

columns=[('RainTomorrow', 'RainToday'),
          ('RainTomorrow', 'Humidity9am'),
          ('RainTomorrow', 'WindGustSpeed'),
          ('RainTomorrow', 'Rainfall')]
for col in columns:
    plt.figure(figsize=(10, 6))
    plt.scatter(x=df[col[0]], y=df[col[1]])
    plt.xlabel(col[0])
    plt.ylabel(col[1])
    plt.title(f'{col[1]}')
    st.pyplot(plt)
