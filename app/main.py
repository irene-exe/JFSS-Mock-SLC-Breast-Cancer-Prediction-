import streamlit as st
import pickle
import pandas as pd 
import plotly.graph_objects as go
import numpy as np

def cleanData():
    data = pd.read_csv("data.csv")
    data = data.drop({"Unnamed: 32", "id"}, axis=1)
    
    data["diagnosis"] = data['diagnosis'].map({'M':1, 'B':0})
    return data

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data = cleanData()
    sliderLabels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]
    
    inputs = {}
    for label, key in sliderLabels:
        inputs[key] = st.sidebar.slider(
            label,
            min_value = 0.0,
            max_value = data[key].max()*1.2,
            value=data[key].mean()
        )
        
    return inputs

def getScaledValues(input_dict):
    data = cleanData()
    x = data.drop(["diagnosis"], axis=1)
    
    scaledDict = {}
    
    for key, value in input_dict.items():
        maxVal = x[key].max()*1.2
        minVal = x[key].min()*0.8
        scaled_value = (value-minVal)/(maxVal-minVal)
        scaledDict[key] = scaled_value
        
    return scaledDict

def getRadarChart(input_data):
  
    input_data = getScaledValues(input_data)
    
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                    'Smoothness', 'Compactness', 
                    'Concavity', 'Concave Points',
                    'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
            r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
            ],
            theta=categories,
            fill='toself',
            name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
            r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
            ],
            theta=categories,
            fill='toself',
            name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
            r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
            ],
            theta=categories,
            fill='toself',
            name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
        showlegend=True
    )
    
    return fig

def addPredictions(input_data):
    model = pickle.load(open("breast-cancer/model.pkl", "rb"))
    scaler = pickle.load(open("breast-cancer/scaler.pkl", "rb"))
    
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    scaled_array = scaler.transform(input_array)
    
    prediction = model.predict(scaled_array)
    
    if (prediction[0]==0): st.write("Benign")
    else: st.write("Malicious")
    
    st.write("Probability of being Benign: ", round(model.predict_proba(scaled_array)[0][0],2))
    st.write("Probability of being Malicious: ", round(model.predict_proba(scaled_array)[0][1], 2))

def main():
    print("Web App")
    
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    inputs = add_sidebar()
    
    with st.container():
        st.title("Breast Cancer Predictor") #h1
        st.write("blah blah blah") #p
        
    col1, col2 = st.columns((4,1), gap="medium")
    with col1:
        radarChart = getRadarChart(inputs)
        st.plotly_chart(radarChart)
    with col2:
        st.subheader("Cell Cluster Prediction")
        addPredictions(inputs)
        
    return 

if __name__ == "__main__":
    main()