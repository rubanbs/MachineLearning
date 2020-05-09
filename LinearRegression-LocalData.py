import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model

stats = np.array([[0,0],[1,0.5],[2,1],[3,1.5],[3.5,1.9],[4,2],[5,2.5],[5.5,2.9],[6,3],[7,3.5],[7.5,3.9],[8,4]])
dataset = pd.DataFrame(stats, columns={'input', 'output'})

X = np.c_[dataset['input']]
y = np.c_[dataset['output']]

# Train linear model
model = sklearn.linear_model.LinearRegression()
model.fit(X, y)

# Predict
new_X = [[3.3]]
new_Y = model.predict(new_X)

dataset.plot(kind="scatter", x="input", y="output")
plt.scatter(new_X, new_Y, color='red')
plt.show()