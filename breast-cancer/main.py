import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

def createModel(data):
    x = data.drop(["diagnosis"], axis=1)
    y = data["diagnosis"]
    
    scaler = StandardScaler()
    
    # scales data
    x = scaler.fit_transform(x)
    
    # splits data
    x_train, x_test, y_train, y_test = train_test_split(
        x,y, test_size=0.2, random_state=42
    )
    
    # training the model
    model = LogisticRegression()
    model.fit(x_train, y_train)
    
    # testing the model
    y_pred = model.predict(x_test)
    print("Model Accuracy: ", accuracy_score(y_test, y_pred))
    print("Classification Report: ", classification_report(y_test, y_pred))
    
    return model, scaler

def cleanData():
    data = pd.read_csv("data.csv")
    data = data.drop({"Unnamed: 32", "id"}, axis=1)
    
    data["diagnosis"] = data['diagnosis'].map({'M':1, 'B':0})
    return data

def main():
    data = cleanData()
    
    model, scaler = createModel(data)
    
    with open("breast-cancer/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("breast-cancer/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    return

if __name__ == '__main__':
    main()