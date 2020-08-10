#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 18:27:31 2020

@author: guillem
"""

import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import plotly.express as px


df = pd.read_csv('~/DadesAirBNB/DatosLimpios.csv')
df['LogGoodprice'] = np.log(df['goodprice'])

app = dash.Dash(__name__)

tabs_styles = {
    'height': '44px',
    'background': '#b50413'
}
tab_style = {
    'borderColor': 'solid #EFFBF8',
    'borderWidth': '2px',
    'color': '#a2010e',
    'background': '#f81023',
    'fontWeight': 'bold',
    'padding': '10px'
}

tab_selected_style = {
    'borderTop': '2px solid #EFFBF8',
    'backgroundColor': '#b80413',
    'color': '#6C0103',
    'fontWeight': 'bold',
    'padding': '10px'
}

app.layout = html.Div(
    [
     html.Div(
             id = 'banner',
             className = 'banner',
             children = [
                 html.H1(
             'ANÁLISIS AIRBNB BARCELONA'),
                ],
             style = {
                 'color': '#FAFAFA',
                 'fontSize': 20,
                 'fontAlign':'center',
                 'fontFamily': 'Brown',
                 'fontVariant': 'small-caps',
                 'padding': '1px',
                 'backgroundColor': '#FC0033',
                 'paddingLeft': '20px',
                 'borderColor': 'black'
                 }
             ),
     
    dcc.Tabs(
            id="tabs-styled-with-inline", 
            value='tab-1', 
            children=[
                dcc.Tab(
                    label = 'Variables Numéricas', 
                    value='tab-1',
                    style=tab_style, 
                    selected_style=tab_selected_style
                    ),
                dcc.Tab(
                    label='Variables Categóricas', 
                    value='tab-2', 
                    style=tab_style, 
                    selected_style=tab_selected_style
                    ),
                dcc.Tab(
                    label='Variables Dicotómicas', 
                    value='tab-3', 
                    style=tab_style, 
                    selected_style=tab_selected_style
                    ),
                dcc.Tab(
                    label='Mapas', 
                    value='tab-4', 
                    style=tab_style, 
                    selected_style=tab_selected_style
                    ),
                dcc.Tab(
                    label='Predictor', 
                    value='tab-5', 
                    style=tab_style, 
                    selected_style=tab_selected_style
                    )
                ], 
            style=tabs_styles
            ),
    html.Div(id='tabs-content-inline')
], style = {'backgroundColor': '#FAFAFA'})


fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    specs=[[{"type": "table"}],
           [{"type": "scatter"}],
           [{"type": "scatter"}]]
)

fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Mining-revenue-USD"],
        mode="lines",
        name="mining revenue"
    ),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Hash-rate"],
        mode="lines",
        name="hash-rate-TH/s"
    ),
    row=2, col=1
)

@app.callback(Output('tabs-content-inline', 'children'),
              [Input('tabs-styled-with-inline', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Variables Numéricas'),
            html.Div([
                dcc.Graph(
                figure={
                    'data': [
                        {'x': df.groupby('month_year')['goodprice'].mean().index, 
                         'y': df.groupby('month_year')['goodprice'].mean(),
                         'type': 'line'},
                            ]
                    }
                    )
                ])
        ])
    elif tab == 'tab-2':
        return html.Div([
             html.Div([
                dcc.Graph(
                figure = fig_tab2)
                 ])
            ], style = {'backgroundColor': '#848484'})
    
    elif tab == 'tab-3':
        return html.Div([
            html.H3('Variables Dicotómicas')
        ])
    elif tab == 'tab-4':
        return html.Div([
            html.H3('Mapas')
        ])
    elif tab == 'tab-5':
        return html.Div([
            html.H3('Predictor')
        ])
    
if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port='8050')