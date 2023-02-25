#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 原版可執行文件，嚴禁更動
from dash import Dash, dcc, Output, Input
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
# 測試用
d = {'year': [2019, 2019, 2019, 2019, 2020, 2020, 2020, 2020, 2021, 2021, 2021, 2021],
     'type': ['老師', '學生', '附屬學生', '家長', '老師', '學生', '附屬學生', '家長', '老師', '學生', '附屬學生', '家長'],
     'value': [1.0074, -2.1533, -1.0595, 0.9708, -0.5849, -1.3441, -0.7611, 0.0895, 0.3272, 0.3204, -1.4274, 1.5207]
    }
df = pd.DataFrame(d)
d = {'students': [60434, 110316, 115272, 173626, 103425, 
              54160, 78977, 94582, 102359, 186365, 
              104225, 93525, 78928, 59904, 99509, 
              33247, 639343, 181445, 173064, 111737, 
              101745, 139556, 97751, 96592, 95807, 
              36120, 105878, 94652, 182419, 54126]}
df2 = pd.DataFrame(d)

x = np.random.normal(10, 0.5, 1000)
y = np.random.normal(20, 0.9, 1000)
z = np.random.normal(30, 8, 1000)
c = np.random.normal(40, 8.5, 1000)

type_list = ['老師', '學生', '附屬帳號', '家長']
year_list = ['2020', '2021', '2022']
xyz_list = ['x', 'y']
app = Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB])

# Declare server for Heroku deployment. Needed for Procfile.
server = app.server

mytitle = dcc.Markdown(children='# 各類帳號年成長率(非真實資料)')
mytitle2 = dcc.Markdown(children='# 三年註冊人數變化(非真實資料)')
mytitle3 = dcc.Markdown(children='# 隨機資料散點圖')
mygraph = dcc.Graph(figure={})
mygraph2 = dcc.Graph(figure={})
mygraph3 = dcc.Graph(figure={})
dropdown = dcc.Dropdown(options=['老師', '學生', '附屬帳號', '家長'],
                       value='老師',
                       clearable=False)
dropdown2 = dcc.Dropdown(options=['2020', '2021', '2022'],
                       value='2021',
                       clearable=False)
dropdown3 = dcc.Dropdown(options=['x', 'y'],
                       value='x',
                       clearable=False)
app.layout = dbc.Container([
    dbc.Row([dbc.Col([mytitle], width=6)], justify='center'),
    dbc.Row([dbc.Col([mygraph], width=6, style={"height": "100%"})], style={"height": "10%"}, justify='center'),
    dbc.Row([dbc.Col([dropdown], width=4)], justify='center'),
    dbc.Row([dbc.Col([mytitle2], width=6)], justify='center'),
    dbc.Row([dbc.Col([mygraph2], width=6, style={"height": "100%"})], style={"height": "10%"}, justify='center'),    
    dbc.Row([dbc.Col([dropdown2], width=4)], justify='center'),
    dbc.Row([dbc.Col([mytitle3], width=6)], justify='center'),
    dbc.Row([dbc.Col([mygraph3], width=6, style={"height": "100%"})], style={"height": "10%"}, justify='center'),
    dbc.Row([dbc.Col([dropdown3], width=4)], justify='center')
], fluid=True)

@app.callback(
    Output(mygraph, component_property='figure'),
    Input(dropdown, component_property='value')
)

def updating_graph(input_):
    fig = go.Figure()
    paras_dict = {}
    width_dict = {}
    
    for i in range(1, 5):
        paras_dict['a'+str(i)] = '#9D9D9D'
        width_dict['w'+str(i)] = 2
    for i in range(4):
        if input_ == type_list[i]:
            paras_dict['a'+str(i+1)] = '#CE0000'
            width_dict['w'+str(i+1)] = 4
            break
    for i in range(4):
        fig.add_trace(go.Scatter(x=[2019, 2020, 2021], y=[df['value'][0+i],
                                                     df['value'][4+i],
                                                     df['value'][8+i]], name=type_list[i],
                     line=dict(color=paras_dict['a'+str(i+1)], width=width_dict['w'+str(i+1)])))
    fig.update_layout(
        xaxis = dict(
            tickmode = 'linear',
            tick0 = 2019,
            dtick = 1
        ),
        margin=dict(b=10, l=10, t=10, r=10),
        height=250,
        plot_bgcolor= '#F0F0F0'
    )
    return fig

@app.callback(
    Output(mygraph2, component_property='figure'),
    Input(dropdown2, component_property='value')
)
def updating_graph2(input_):
    fig2 = go.Figure()
    paras_dict2 = {}
    width_dict2 = {}
    for i in range(1, 4):
        paras_dict2['a'+str(i)] = '#9D9D9D'
        width_dict2['w'+str(i)] = 2
    for i in range(3):
        if input_ == year_list[i]:
            paras_dict2['a'+str(i+1)] = '#CE0000'
            width_dict2['w'+str(i+1)] = 4
            break
    for year in range(3):
        if year != 2:
            fig2.add_trace(go.Scatter(x=[i for i in range(1, 13)], 
                                     y=df2['students'][(year*12):(year+1)*12], 
                                     name=year_list[year],
                     line=dict(color=paras_dict2['a'+str(year+1)], width=width_dict2['w'+str(year+1)])))
        else:
            fig2.add_trace(go.Scatter(x=[i for i in range(1, 7)], 
                                 y=df2['students'][6:], 
                                 name=year_list[year],
                 line=dict(color=paras_dict2['a'+str(year+1)], width=width_dict2['w'+str(year+1)])))
    fig2.update_layout(
        xaxis = dict(
            tickmode = 'linear',
            tick0 = 1,
            dtick = 1
        ),
        margin=dict(b=10, l=10, t=10, r=10),
        height=250,
        plot_bgcolor= '#F0F0F0'
    )
    return fig2
@app.callback(
    Output(mygraph3, component_property='figure'),
    Input(dropdown3, component_property='value')
)
def updating_scatter(input_):
    fig3 = go.Figure()
    paras_dict3 = {}
    if input_ == 'x':
        fig3.add_trace(
            go.Scatter(
                x=x, y=y, 
                mode='markers', 
                marker=dict(
                    size=z, 
                    color=c, 
                    colorscale='Viridis', 
                    showscale=True)))
    else:
        fig3.add_trace(
            go.Scatter(
               x=y, y=z,
               mode='markers',
               marker=dict(
                   size=c,
                   color=x, #set color equal to a variable
                   colorscale='Viridis', # one of plotly colorscales
                   showscale=True
                   )))
    fig3.update_layout(
        margin=dict(b=10, l=10, t=10, r=10),
        height=450,
        width=1300,
        plot_bgcolor= '#F0F0F0'
    )
    return fig3

if __name__ == '__main__':
    app.run_server()


# In[ ]:




