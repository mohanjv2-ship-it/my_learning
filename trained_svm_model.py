import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

df = load_breast_cancer()
x = df.data
y = df.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)

model = SVC(kernel="linear")
mod = model.fit(x_train,y_train)

ypred = mod.predict(x_test)
print("Accuracy_Score:", accuracy_score(y_test, ypred))

#Save model and scale
pickle.dump(model, open("svc_model.pkl", "wb"))
pickle.dump(scale, open("scale.pkl", "wb"))

print("Model and Scaler sSaved Successfully!")