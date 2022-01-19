import plotly.express as px
import numpy as np

def readFile(fileName):
        fileObj = open(fileName, "r") #opens the file in read mode
        words = fileObj.read().splitlines() #puts the file into an array
        fileObj.close()
        return words
 
filename = 'LTG_cs_per_min.txt'

lines = readFile(filename)
xsize = [i for i in range(0, len(lines))]

df = px.data.tips()
fig = px.scatter(df, x=xsize, y=np.array(lines).astype(np.float64), trendline="ols")

fig.show()
