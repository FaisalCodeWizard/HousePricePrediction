from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

def home(request):
    return render(request, "home.html")

def predict(request):
    return render(request, "predict.html")

def result(request):
    # Use a raw string for the file path
    df = pd.read_csv(r"C:\Users\hp\Downloads\Housing.csv")

    # Converting categorical data into numerical data
    varlist = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    df[varlist] = df[varlist].apply(lambda x: x.map({'yes': 1, "no": 0}))

    df['furnishingstatus'] = df['furnishingstatus'].replace({'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0})

    # Dropping unrelated columns
    df = df.drop(['bathrooms', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea'], axis='columns')

    # Creating a new feature "price_per_sqft"
    df['price_per_sqft'] = df['price'] / df['area']

    # Splitting the data into training and testing sets
    X = df.drop('price', axis='columns')
    Y = df['price']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

    # Selecting 'Random Forest' as it gives the highest best scores
    model = RandomForestRegressor()
    model.fit(X_train,Y_train)
    model.score(X_test,Y_test)

    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    cross_val_score(RandomForestRegressor(), X, Y, cv=cv)

    # Input values from the request
    var1 = float(request.GET.get('n1', 0))
    var2 = float(request.GET.get('n2', 0))
    var3 = float(request.GET.get('n3', 0))
    var4 = float(request.GET.get('n4', 0))
    var5 = float(request.GET.get('n5', 0))
    var6 = float(request.GET.get('n6', 0))

    # Predicting the price
    pred = model.predict(np.array([[var1, var2, var3, var4, var5, var6]]))
    pred = round(pred[0])
    
    price = f"The predicted price is ₹ {pred}"

    return render(request, "predict.html", {"result2": price})
