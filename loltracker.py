from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def readFile(fileName):
        fileObj = open(fileName, "r") #opens the file in read mode
        coords = fileObj.read().splitlines() #puts the file into an array
        fileObj.close()
        return coords
 
filename = 'LTG_cs_per_min.txt'

lines = readFile(filename)
xsize = [i for i in range(0, len(lines))]

df = pd.DataFrame(
    data={'CS_per_min':np.array(lines).astype(np.float64), 'Volume':np.array(lines).astype(np.float64)}, 
    index=xsize
)
X = np.array(xsize).reshape(-1,1)

model = LinearRegression()
model.fit(X, np.array(lines).astype(np.float64))

x_range = np.linspace(X.min(), X.max(), 100)
y_range = model.predict(x_range.reshape(-1, 1))

fig = make_subplots(rows=1, cols=2)

fig.add_trace(
    go.Scatter(x=df.index, y=df.CS_per_min, name='CS per min' ),
    row=1, col=1
)
fig.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))

fig.add_trace(
    go.Scatter(x=df.index, y=df.Volume, name='Volume'),
    row=1, col=2
)
fig.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))

fig.show()