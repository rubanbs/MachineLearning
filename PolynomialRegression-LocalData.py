import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

n_samples = 100
X = np.linspace(0, 10, 100)
y = X ** 3 + np.random.randn(n_samples) * 100 + 100
plt.figure(figsize=(10,8))
plt.scatter(X, y)

# LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X.reshape(-1, 1), y)
model_pred = lin_reg.predict(X.reshape(-1,1))
#plt.plot(X, model_pred)
print(r2_score(y, model_pred))

# Polynomial Regression ^2
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X.reshape(-1, 1))
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y.reshape(-1, 1))
model_pred_poly = lin_reg_2.predict(X_poly)
#plt.plot(X, model_pred_poly, color='red')
print(r2_score(y, model_pred_poly))

# Polynomial Regression ^3
poly_reg_3 = PolynomialFeatures(degree=3)
X_poly_3 = poly_reg_3.fit_transform(X.reshape(-1, 1))
lin_reg_3 = LinearRegression()
lin_reg_3.fit(X_poly_3, y.reshape(-1, 1))
model_pred_poly_3 = lin_reg_3.predict(X_poly_3)
plt.plot(X, model_pred_poly_3, color='green')
print(r2_score(y, model_pred_poly_3))

plt.show()
