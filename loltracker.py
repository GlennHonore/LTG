import plotly.express as px
 
 
# Creating the Figure instance
df = px.data.tips()
fig = px.scatter(df, x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], y=[1, 2, 3, 5, 1, 6, 8, 4, 2, 11], trendline="ols")

with open('readme.txt') as f:
    lines = f.readlines()
    
# showing the plot
fig.show()