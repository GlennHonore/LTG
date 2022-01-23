from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def addGameMenu():
    filename = ['LTG_damage_per_gold.txt', 'LTG_early_gold_adv.txt', 'LTG_early_cs_adv.txt', 'LTG_cs_per_min.txt']
    dataCoords = []
    dataCoords.append(input('Damages per gold : '))
    dataCoords.append(input('Early gold advantage : '))
    dataCoords.append(input('Early CS advantage : '))
    dataCoords.append(input('CS per min : '))
    print(dataCoords)
    for i in range(0, len(filename)):
        f = open(filename[i], "a")
        f.write("\n"+dataCoords[i])
        f.close()


def readFile(fileName):
    fileObj = open(fileName, "r")
    coords = fileObj.read().splitlines()
    fileObj.close()
    return coords
 
def calculateRegression(xsize, index, lines, model):
    X = np.array(xsize).reshape(-1,1)
    model.fit(X, np.array(lines[index]).astype(np.float64))
    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))
    return x_range, y_range

def setupData(lines, xsize):

    df = pd.DataFrame(data={
        'Damage_per_gold':np.array(lines[0]).astype(np.float64), 
        'Early_gold_adv':np.array(lines[1]).astype(np.float64),
        'Early_cs_adv':np.array(lines[2]).astype(np.float64),
        'CS_per_min':np.array(lines[3]).astype(np.float64)}, 
    index=xsize)
    return df

def linesValues():
    filename = ['LTG_damage_per_gold.txt', 'LTG_early_gold_adv.txt', 'LTG_early_cs_adv.txt', 'LTG_cs_per_min.txt']
    lines = []

    for i in range(0, len(filename)):
        lines.insert(i, readFile(filename[i]))
    return lines

def makeGraphes():
    model = LinearRegression()
    lines = linesValues()
    xsize = [j for j in range(0, len(lines[0]))]
    df = setupData(lines, xsize)
    fig = make_subplots(rows=2, cols=3)

    x_range, y_range = calculateRegression(xsize, 0, lines, model)
    fig.add_trace(
        go.Scatter(x=df.index, y=df.Damage_per_gold, name='Damage per gold'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=x_range, y=y_range, name='Regression - Damage per gold'),
        row=1, col=2
    )

    x_range, y_range = calculateRegression(xsize, 1, lines, model)
    fig.add_trace(
        go.Scatter(x=df.index, y=df.Early_gold_adv, name='Early gold advantage' ),
        row=1, col=3
    )
    fig.add_trace(
        go.Scatter(x=x_range, y=y_range, name='Regression - Early gold adv'), 
        row=1, col=3
    )

    x_range, y_range = calculateRegression(xsize, 2, lines, model)
    fig.add_trace(
        go.Scatter(x=df.index, y=df.Early_cs_adv, name='Early CS advantage' ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_range, y=y_range, name='Regression - Early CS adv'), 
        row=2, col=1
    )

    x_range, y_range = calculateRegression(xsize, 3, lines, model)
    fig.add_trace(
        go.Scatter(x=df.index, y=df.CS_per_min, name='CS per min' ),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=x_range, y=y_range, name='Regression - CS per min'), 
        row=2, col=2
    )

    fig.show()

while 1:
    option = input('1 | Add new game\n'
    '2 | generate graphs\n'
    '3 | exit\n')
    if option == "1":
        addGameMenu()
    if option == "2":
        makeGraphes()
    if option == "3":
        exit()
