# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 23:05:16 2023

@author: debna
"""

import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

#load the model
model = pickle.load(open('forecast_model_doubleexp.pickle','rb'))

#load dataset to plot alongside predictions
df = pd.read_csv("DayForecast.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index(['Date'], inplace=True)

#page configuration
st.set_page_config(layout='centered')
image = Image.open('E:/LiveProject/Medical_drug_forecast/Model_Deployment/download.jpg')
st.image(image)

date = st.slider("Select number of dates",1,60,step = 1)
    
    
pred = model.forecast(date)
pred = pd.DataFrame(pred, columns=['Quantity'])
   
if st.button("Predict"):

        col1, col2 = st.columns([2,3])
        with col1:
             st.dataframe(pred)
        with col2:
            fig, ax = plt.subplots()
            df['Quantity'].plot(style='--', color='gray', legend=True, label='known')
            pred['Quantity'].plot(color='b', legend=True, label='prediction')
            st.pyplot(fig)
