# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import os
import dash
import glob
from pathlib import Path
import numpy as np
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from algos import method1
from algos import method2
from algos import method3
from algos import method4

external_stylesheets = ['app.css']

def sort_list(_list):
    _list = [int(i) for i in _list]
    _list.sort()
    _list = [str(i) for i in _list]
    return _list

def parse_time(x):
    t = x
    if x[0] == '-':
        t = x.replace('-', '').strip()
    t = t.split(':')
    t = float(t[0]) * 60 + float(t[1])
    if x[0] == '-':
        t *= -1
    return t

def csv_reader(filename):
    data=pd.read_csv('./cardio/{}.csv'.format(filename),names=['time','HR','State'],skiprows=1)
    data.head(5)
    data.drop('State', axis=1, inplace=True)
    data = data[~data['time'].str.contains('-')]
    data['time'] = data['time'].apply(lambda x: parse_time(x))
    data.dropna(inplace=True)
    data.index = pd.RangeIndex(start=0, stop=len(data), step=1)
    return data

def preprocess(df, intervalle_slider, threshold_slider, threshold_mode, wavelet_mode, waverdec_mode):
    data1 = method1(df, intervalle_slider)
    df['HR-method1-mean'] = data1['HR_mean']
    df['HR-method1-median'] = data1['HR_median']
    data2 = method2(df, intervalle_slider)
    df['HR-method2'] = data2['HR_modified']
    data3 = method3(df)
    df['HR-method3'] = data3['HR-method3']
    data4 = method4(df, threshold_slider, threshold_mode, wavelet_mode, waverdec_mode)
    df['HR-method4'] = data4['HR-method4']
    df.rename(columns={'HR': 'HR-original'}, inplace= True)
    return df

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

available_indicators = sort_list([Path(p).stem for p in glob.glob('cardio/*.csv')])
thresh_mode = ['soft', 'hard', 'garrote', 'greater', 'less']
wavelet_mode = ['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38', 'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20', 'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8', 'coif9', 'coif10', 'coif11', 'coif12', 'coif13', 'coif14', 'coif15', 'coif16', 'coif17', 'dmey', 'bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8', 'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8']
waverdec_mode = ['zpd', 'cpd', 'sym', 'ppd', 'sp1', 'per']


app.layout = html.Div([
    html.H1(
        children="Aperçu des différentes méthodes de lissage des enregistrements",
        style={'textAlign': 'center'}
    ),
    html.Div([
        html.Div([
            html.Label('Fichier csv'),
            dcc.Dropdown(
                id='filename',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='42',
                clearable=False
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.Label('Méthodes'),
            dcc.Checklist(
                id='methods-used',
                options=[
                    {'label': 'Original', 'value': 'original'},
                    {'label': 'Méthode 1', 'value': 'method1'},
                    {'label': 'Méthode 2', 'value': 'method2'},
                    {'label': 'Méthode 3', 'value': 'method3'},
                    {'label': 'Méthode 4', 'value': 'method4'}
                ],
                value=['original']
            ),
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),
    dcc.Graph(id='indicator-graphic'),
    html.Div([
        html.H3('Méthode 1 & 2')
        ], style={'width': '10%', 'display': 'inline-block'}
    ),
    html.Div([
        html.Label('Intervalle'),
        dcc.Slider(
        id='intervalle-slider',
        min=1,
        max=1000,
        value=30,
        marks={str(k) : str(k) for k in range(0, 1001, 100)},
        step=1
        )], style={'width': '90%', 'display': 'inline-block'}
    ),
    html.Div([
        html.H3('Méthode 4')
        ], style={'width': '10%', 'display': 'inline-block'}
    ),
    html.Div([
            html.Label('Paramètre 1 : threshold mode'),
            dcc.Dropdown(
                id='threshold-mode',
                options=[{'label': i, 'value': i} for i in thresh_mode],
                value='soft',
                clearable=False
            ),
        ], style={'width': '15%', 'display': 'inline-block'}),
    html.Div([
            html.Label('Paramètre 2 : wavelet mode'),
            dcc.Dropdown(
                id='wavelet-mode',
                options=[{'label': i, 'value': i} for i in wavelet_mode],
                value='db8',
                clearable=False
            ),
        ], style={'width': '15%', 'display': 'inline-block'}),
    html.Div([
            html.Label('Paramètre 3 : wavedec & waverec mode'),
            dcc.Dropdown(
                id='waverdec-mode',
                options=[{'label': i, 'value': i} for i in waverdec_mode],
                value='per',
                clearable=False
            ),
        ], style={'width': '20%', 'display': 'inline-block'}),
    html.Div([
        html.Label('Threshold'),
        dcc.Slider(
        id='threshold-slider',
        min=0,
        max=1,
        value=0.63,
        marks={str(k/10) : str(k/10) for k in range(0, 10, 1)},
        step=0.01
        )], style={'width': '40%', 'float': 'right', 'display': 'inline-block'}),
    html.Div(id= 'table'),
    html.Div([
        html.H2("Description des paramètres"),
    ], style={'textAlign' : 'center'}),
    html.Div([
        html.H3("Intervalle"),
        html.P("Désigne n dans le calcul des moyennes mobiles (n + valeur + n).")
    ], style={'textAlign' : 'left'}),
    html.Div([
        html.H3("Threshold"),
        html.P("Désigne le seuil à partir duquel les valeurs vont être modifiées.")
    ], style={'textAlign' : 'left'}),
    html.Div([
        html.H3("Paramètre 1 : threshold mode"),
        dcc.Link(href= "https://pywavelets.readthedocs.io/en/latest/ref/thresholding-functions.html"),
        html.H4("Soft:"),
        html.P("Data values with absolute value less than param are replaced with substitute. Data values with absolute value greater or equal to the thresholding value are shrunk toward zero by value. In other words, the new value is data/np.abs(data) * np.maximum(np.abs(data) - value, 0)."),
        html.H4("Hard:"),
        html.P("The data values where their absolute value is less than the value param are replaced with substitute. Data values with absolute value greater or equal to the thresholding value stay untouched."),
        html.H4("Garrot:"),
        html.P("Corresponds to the Non-negative garrote threshold. It is intermediate between hard and soft thresholding. It behaves like soft thresholding for small data values and approaches hard thresholding for large data values."),
        html.H4("Greater:"),
        html.P("The data is replaced with substitute where data is below the thresholding value. Greater data values pass untouched."),
        html.H4("Less:"),
        html.P("The data is replaced with substitute where data is above the thresholding value. Lesser data values pass untouched.")
    ], style={'textAlign' : 'left'}),
    html.Div([
        html.H3("Paramètre 2 : wavelet mode"),
        dcc.Link(href= "https://www.mathworks.com/help/wavelet/gs/introduction-to-the-wavelet-families.html"),
        html.H4("db - Daubechies"),
        html.P("The dbN wavelets are the Daubechies’ extremal phase wavelets. N refers to the number of vanishing moments. These filters are also referred to in the literature by the number of filter taps, which is 2N."),
        html.H4("Symlet Wavelets: symN"),
        html.P("The symN wavelets are also known as Daubechies’ least-asymmetric wavelets. The symlets are more symmetric than the extremal phase wavelets. In symN, N is the number of vanishing moments. These filters are also referred to in the literature by the number of filter taps, which is 2N."),
        html.H4("Coiflet Wavelets: coifN"),
        html.P("Coiflet scaling functions also exhibit vanishing moments. In coifN, N is the number of vanishing moments for both the wavelet and scaling functions. These filters are also referred to in the literature by the number of filter coefficients, which is 3N."),
        html.H4("dmey"),
        html.P("“Discrete” FIR approximation of Meyer wavelet, which is :"),
        html.P("Both ψ and φ are defined in the frequency domain, starting with an auxiliary function ν"),
        html.H4("Biorthogonal Wavelet Pairs: biorNr.Nd"),
        html.P("Biorthogonal wavelets feature a pair of scaling functions and associated scaling filters — one for analysis and one for synthesis."),
        html.P("There is also a pair of wavelets and associated wavelet filters — one for analysis and one for synthesis."),
        html.P("The analysis and synthesis wavelets can have different numbers of vanishing moments and regularity properties. You can use the wavelet with the greater number of vanishing moments for analysis resulting in a sparse representation, while you use the smoother wavelet for reconstruction."),
        html.H4("rbio"),
        html.P("Reverse biorthogonal."),
            ], style={'textAlign' : 'left'}),
    html.Div([
        html.H3("Paramètre 3 : wavedec & waverec mode"),
        dcc.Link(href= "https://pywavelets.readthedocs.io/en/0.2.2/ref/signal-extension-modes.html#ref-modes"),
        html.H4("zpd - zero-padding"),
        html.P("signal is extended by adding zero samples:"),
        html.P("... 0  0 | x1 x2 ... xn | 0  0 ..."),
        html.H4("cpd - constant-padding"),
        html.P("border values are replicated:"),
        html.P("... x1 x1 | x1 x2 ... xn | xn xn ..."),
        html.H4("sym - symmetric-padding"),
        html.P("signal is extended by mirroring samples:"),
        html.P("... x2 x1 | x1 x2 ... xn | xn xn-1 ..."),
        html.H4("ppd - periodic-padding"),
        html.P("signal is treated as a periodic one:"),
        html.P("... xn-1 xn | x1 x2 ... xn | x1 x2 ..."),
        html.H4("sp1 - smooth-padding"),
        html.P("signal is extended according to the first derivatives calculated on the edges (straight line)"),
        html.H4("DWT"),
        html.P("performed for these extension modes is slightly redundant, but ensures perfect reconstruction. To receive the smallest possible number of coefficients, computations can be performed with the periodization mode:"),
        html.H4("per - periodization"),
        html.P("is like periodic-padding but gives the smallest possible number of decomposition coefficients. IDWT must be performed with the same mode."),
    ], style={'textAlign' : 'left'})
])

@app.callback(
    [Output('indicator-graphic', 'figure'),
    Output('table', 'children')],
    [Input('filename', 'value'),
    Input('methods-used', 'value'),
    Input('intervalle-slider', 'value'),
    Input('threshold-slider', 'value'),
    Input('threshold-mode', 'value'),
    Input('wavelet-mode', 'value'),
    Input('waverdec-mode', 'value')]
)

def update_graph(filename, methods_used, intervalle_slider, threshold_slider, threshold_mode, wavelet_mode, waverdec_mode):
    dff = preprocess(csv_reader(filename), intervalle_slider, threshold_slider, threshold_mode, wavelet_mode, waverdec_mode)
    fig = go.Figure()
    colors = px.colors.qualitative.Dark24
    for method in methods_used :
        if method == 'original' :
            fig.add_trace(go.Scatter(x=dff['time'], y=dff['HR-{}'.format(method)], mode='lines', name='graph-{}'.format(method), line=dict(color=colors[0])))
        if method == 'method1' :
            fig.add_trace(go.Scatter(x=dff['time'], y=dff['HR-{}-mean'.format(method)], mode='lines', name='graph-{}-mean'.format(method), line=dict(color=colors[1])))
            fig.add_trace(go.Scatter(x=dff['time'], y=dff['HR-{}-median'.format(method)], mode='lines', name='graph-{}-median'.format(method), line=dict(color=colors[2])))
        if method == 'method2' :
            fig.add_trace(go.Scatter(x=dff['time'], y=dff['HR-{}'.format(method)], mode='lines', name='graph-{}'.format(method), line=dict(color=colors[3])))
        if method == 'method3' :
            fig.add_trace(go.Scatter(x=dff['time'], y=dff['HR-{}'.format(method)], mode='lines', name='graph-{}'.format(method), line=dict(color=colors[4])))
        if method == 'method4' :
            fig.add_trace(go.Scatter(x=dff['time'], y=dff['HR-{}'.format(method)], mode='lines', name='graph-{}'.format(method), line=dict(color=colors[5])))

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 250}, hovermode='closest', showlegend=True)
    fig.update_yaxes(title='Patient n°' + str(filename))
    fig.update_xaxes(title='Temps (s)')

    dfff = dff.describe()
    index = ['0', '1', '2', '3', '4', '5', '6', '7']
    dfff.index = index
    eoifeij = pd.DataFrame(index=index)
    eoifeij['stats'] = ['temps total (s)', 'moyenne', 'écart-type', 'minimum', '1er quartile : 25%', 'médiane', '3eme quartile : 75%', 'maximum']
    d = pd.concat([eoifeij, dfff], axis = 1)
    del d['time']

    return fig, dash_table.DataTable(data= d.to_dict('records'), columns= [{"name": i, "id": i} for i in d.columns])

if __name__ == '__main__':
    app.run_server(debug=True)
