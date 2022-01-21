from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def readFile(fileName):
    fileObj = open(fileName, "r")
    coords = fileObj.read().splitlines()
    fileObj.close()
    return coords
 
def calculateRegression(xsize, index):
    X = np.array(xsize).reshape(-1,1)
    model.fit(X, np.array(lines[index]).astype(np.float64))
    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))
    return x_range, y_range


filename = ['LTG_damage_per_gold.txt', 'LTG_cs_per_min.txt']

model = LinearRegression()
lines = []

for i in range(0, len(filename)):
    lines.insert(i, readFile(filename[i]))
    
xsize = [j for j in range(0, len(lines[0]))]

df = pd.DataFrame(data={
    'Damage_per_gold':np.array(lines[0]).astype(np.float64), 
    'CS_per_min':np.array(lines[1]).astype(np.float64)}, 
    index=xsize)

x_range, y_range = calculateRegression(xsize, 0)

fig = make_subplots(rows=1, cols=2)
fig.add_trace(
    go.Scatter(x=df.index, y=df.Damage_per_gold, name='Damage per gold'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=x_range, y=y_range, name='Regression Fit'),
    row=1, col=1
)

x_range, y_range = calculateRegression(xsize, 1)

fig.add_trace(
    go.Scatter(x=df.index, y=df.CS_per_min, name='CS per min' ),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(x=x_range, y=y_range, name='Regression Fit'), 
    row=1, col=2
)
fig.show()