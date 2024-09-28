import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import Ridge,Lasso

df = pd.read_csv(r"C:\Users\jayas\Downloads\Ds Datasets\emp_sal.csv")

x = df.iloc[:,1:2].values
y = df.iloc[:,2].values

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x,y)

plt.scatter(x,y,color = 'red')
plt.plot(x,reg.predict(x),color = 'blue')

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)

poly_reg.fit(x_poly,y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

from sklearn.svm import SVR
reg1 = SVR(kernel= "poly",degree=6)
reg1.fit(x,y)

from sklearn.tree import DecisionTreeRegressor
reg_tree = DecisionTreeRegressor()
reg_tree.fit(x,y)
tree_pred = reg.predict([[6.5]])


plt.scatter(x,y,color = 'red')
plt.plot(x,lin_reg_2.predict(x_poly),color = 'blue')

new_pred = reg.predict([[6.5]])

ridge1 = Ridge()
ridge1.fit(x,y)

ridge1.score(x,y)

lasso_mod = Lasso()
lasso_mod.fit(x,y)

lasso_mod.score(x,y)

filename = "LLM_Model.pkl"
with open(filename, 'wb') as file:
    pickle.dump(reg,file)
