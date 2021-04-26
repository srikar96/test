import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
import requests
import json
import plotly
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output, State
from pandas.io.json import json_normalize
import time
import random
import string
import flask
import plotly.graph_objs as go
import plotly.express as px

import pandas as pd
import pickle
import numpy as np
# import keras
import matplotlib.pyplot as plt
import re
import string
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_curve, roc_auc_score, auc
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import make_union
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import sklearn.metrics as mk
from googleapiclient import discovery
import json
import config

API_KEY = config.api_key

client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=API_KEY,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)
google_names = ['TOXICITY', 'SEVERE_TOXICITY', 'THREAT', 'INSULT', 'IDENTITY_ATTACK', 'OBSCENE']
# import warnings
# warnings.filterwarnings("ignore")

### Models
mod = ['svm', 'logistic', 'nb']
cats = ['toxic', 'severe_toxic', 'identity_hate', 'threat', 'obscene', 'insult']
# metrics
metrics = {}
for val in mod:
    with open('models/{}_json.pk'.format(val), 'rb') as file:
        metrics[val] = pickle.load(file)

# Vectorizer
with open('models/vectorizer.pk', 'rb') as file:
    vectorizer = pickle.load(file)

# Vectorized testing comments
with open('models/testing_comments_vec.pk', 'rb') as file:
    testing_comments = pickle.load(file)

# y_test
y_test = pd.read_csv('models/y_test.csv')

# models
models = {}
for val in mod:
    models[val] = {}
    for cat in cats:
        with open('models/{}_{}_model.pk'.format(val,cat), 'rb') as file:
            models[val][cat] = pickle.load(file)


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def indicator(color, text, id_value):
    '''
    Input: display Div attributes
    returns the Div
    '''
    return html.Div(
        [

            html.P(
                text,
                className="text-center",
                style={'color':'#f67f7d', 'font-weight':'bold', 'font-size': '180%'}
            ),
            html.P(
                id = id_value,
                className="text-center text-primary",
                style = {'font-size': '180%'}
            )
        ],
        className="col-md-3 col-sm-3 col-6 card card-body",

    )

def indicator1(color, text, id_value):
    '''
    Input: display Div attributes
    returns the Div
    '''
    return html.Div(
        [
            html.P(
                text,
                className="text-center",
                style={'color':'#f67f7d', 'font-weight':'bold', 'font-size': '180%'}
            ),
            html.P(
                id = id_value,
                className="text-center text-primary",
                style = {'font-size': '180%'}
            ),
            # html.progress(id="file", value="32", max="100"),
            html.P(
                className="text-center text-primary",
                style = {'font-size': '80%'}
            )
        ],
        className="col-md-4 col-sm-4 col-6 card card-body",
    )

# Page Layout
app.layout = html.Div([
    html.Div(id='intermediate-value', style={'display': 'none'}),
    html.Div([dcc.Location(id='url', refresh=False), html.Div(id='page-content')]),

    html.Div(id='intermediate-value-consoli', style={'display': 'none'}),

    html.Div(
            [
    # School selection
        html.Div(
            dcc.Dropdown(
                id="school_selection",
                options=[
                    {"label": "SVM", "value": "svm"},
                    {"label": "Logistic Regression", "value": "logistic"},
                    {"label": "Naive Bayes", "value": "nb"}
                ],
                value = 'svm',
                clearable=False,
                placeholder = 'Select Model',
                style={'color': 'black', 'font-size': '120%'}
            ),
            className="col-md-4 col-sm-6 col-4",
        ),
    # Gender selection
        html.Div(
            dcc.Dropdown(
                id="gender_selection",
                options=[
                    {"label": "-", "value": "male"},
                    {"label": "=", "value": "female"},
                    {"label": "~", "value": "all"}

                ],
                value="all",
                clearable=False,
                placeholder = 'Select Param1'
            ),
            className="col-md-2 col-sm-6 col-4",
        ),
    #Class selection
      html.Div(
            dcc.Dropdown(
                id="class_selection",
                value = 'all',
                clearable=False,
                placeholder = 'Select Param2'
            ),
            className="col-md-2 col-sm-6 col-4",
        ),
        # Section selection
        html.Div(
            dcc.Dropdown(
                id="section_selection",
                value="all",
                clearable=False,
                placeholder = 'Select Param3'
            ),
            className="col-md-2 col-sm-6 col-4",
        ),

    ],
    className="row mt-2",
),

    # Display bar Divs
    html.Div(
        [
            indicator1(
                "#119DFF", "Test Accuracy", "first_indicator"
            ),
            indicator1(
                "#EF553B", "F1 Score", "second_indicator",
            ),
            indicator1(
                "#119DFF", "ROC Score", "third_indicator"
            ),
        ],
        className="row mt-2",
    ),
    # charts row div for text and toxic display bars
    html.Div(
        [
            html.Div(
                [
                    html.P(''),
                    html.P(''),
                    dcc.Loading(
                    id="loading-2",
                    type="default",
                    children=html.Div(id="loading-output-2")
                    )
                ],
                className="col-md-6 col-sm-12 text-center text-primary card card-body bg-light"
            ),

            indicator(
                "#119DFF", "Toxic", "down_one_indicator"
            ),
            indicator(
                "#119DFF", "Severe Toxic", "down_two_indicator"
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                            dcc.Input(
                                id="input_text",
                                type="text",
                                placeholder="Type Comment...",
                                style={'font-size':'20px'},
                                className="col-md-12"
                                ),
                            className="col-md-8"),
                            html.Div(
                                dbc.Button("Check", color="primary", className="col-md-12", id="check_button"),
                                className="col-md-4"
                                )
                        ],
                        className="row mt-2"
                    ),
                ],
                className="col-md-6 col-sm-12 text-center text-primary card card-body bg-light"
            ),
            indicator(
                "#119DFF", "Threat", "down_three_indicator"
            ),
            indicator(
                "#119DFF", "Insult", "down_four_indicator"
            ),
            html.Div(
                [   html.P(''),
                    html.P(''),
                    dcc.Loading(
                    id="loading-1",
                    type="default",
                    children=html.Div(id="loading-output-1")
                ),
                ],
                className="col-md-6 col-sm-12 text-center text-primary card card-body bg-light"
            ),
            indicator(
                "#119DFF", "Identity Hate", "down_five_indicator"
            ),
            indicator(
                "#119DFF", "Obscene", "down_six_indicator"
            )
        ],
        className="row mt-2",
    ),
    html.Div(
    [
        html.Div(
                dcc.Graph(
                    id="bar_chart",
                    style={"height": 410},
                    config=dict(displayModeBar=False),
                ),
                className="col-md-5 col-sm-12 text-center text-primary card card-body bg-light"
            ),
        html.Div([
                    dcc.Dropdown(
                        id = "color_selection",
                        options = [{'label':i.capitalize(), 'value':i} for i in cats],
                        value = 'toxic',
                        clearable = False,
                        placeholder = 'Select Category',
                        style={'color': 'black', 'font-size': '120%'}
                    ),
                    dcc.Graph(
                        id="roc_chart",
                        style={"height": 410},
                        config=dict(displayModeBar=False),
                    ),
            ], className="col-md-7 col-sm-12 text-center text-primary card card-body bg-light"),
    ],
    className="row mt-2"
    )
])

@app.callback(
    Output("first_indicator", "children"),
    [Input('school_selection', 'value')])
def accuracy_update(model):
    acc = 0
    for cat in cats:
        acc += metrics[model][cat]['accuracy']
    acc = round(acc*100/6,3)
    return acc

@app.callback(
    Output("second_indicator", "children"),
    [Input('school_selection', 'value')])
def f1_update(model):
    f1 = 0
    for cat in cats:
        f1 += metrics[model][cat]['f1']
    f1 = round(f1*100/6,3)
    return f1

@app.callback(
    Output("third_indicator", "children"),
    [Input('school_selection', 'value')])
def roc_update(model):
    roc = 0
    for cat in cats:
        roc += metrics[model][cat]['roc']
    roc = round(roc*100/6,3)
    return roc

@app.callback(
    Output("loading-output-1", "children"),
    Input("input_text", "value"))
def input_triggers_spinner(value):
    time.sleep(1)
    return value

@app.callback(
    Output("loading-output-2", "children"),
    Input("input_text", "value"))
def input_triggers_spinner1(value):
    time.sleep(1)
    return value

@app.callback(
    Output("down_one_indicator", "children"),
    [Input("check_button", "n_clicks"), Input('school_selection', 'value')],
    state=[State(component_id='input_text', component_property='value')])
def toxic_update(click, model, text):
    if text:
        tst = vectorizer.transform([text])
        if model == 'nb':
            return round(models[model]['toxic'].predict_proba(tst)[0][1], 3)
        else:
            return round(models[model]['toxic']._predict_proba_lr(tst)[0][1], 3)

@app.callback(
    Output("down_two_indicator", "children"),
    [Input("check_button", "n_clicks"), Input('school_selection', 'value')],
    state=[State(component_id='input_text', component_property='value')])
def severe_toxic_update(click, model, text):
    if text:
        tst = vectorizer.transform([text])
        if model == 'nb':
            return round(models[model]['severe_toxic'].predict_proba(tst)[0][1], 3)
        else:
            return round(models[model]['severe_toxic']._predict_proba_lr(tst)[0][1], 3)

@app.callback(
    Output("down_three_indicator", "children"),
    [Input("check_button", "n_clicks"), Input('school_selection', 'value')],
    state=[State(component_id='input_text', component_property='value')])
def threat_update(click, model, text):
    if text:
        tst = vectorizer.transform([text])
        if model == 'nb':
            return round(models[model]['threat'].predict_proba(tst)[0][1], 3)
        else:
            return round(models[model]['threat']._predict_proba_lr(tst)[0][1], 3)

@app.callback(
    Output("down_four_indicator", "children"),
    [Input("check_button", "n_clicks"), Input('school_selection', 'value')],
    state=[State(component_id='input_text', component_property='value')])
def insult_update(click, model, text):
    if text:
        tst = vectorizer.transform([text])
        if model == 'nb':
            return round(models[model]['insult'].predict_proba(tst)[0][1], 3)
        else:
            return round(models[model]['insult']._predict_proba_lr(tst)[0][1], 3)

@app.callback(
    Output("down_five_indicator", "children"),
    [Input("check_button", "n_clicks"), Input('school_selection', 'value')],
    state=[State(component_id='input_text', component_property='value')])
def identity_hate_update(click, model, text):
    if text:
        tst = vectorizer.transform([text])
        if model == 'nb':
            return round(models[model]['identity_hate'].predict_proba(tst)[0][1], 3)
        else:
            return round(models[model]['identity_hate']._predict_proba_lr(tst)[0][1], 3)

@app.callback(
    Output("down_six_indicator", "children"),
    [Input("check_button", "n_clicks"), Input('school_selection', 'value')],
    state=[State(component_id='input_text', component_property='value')])
def obscene_update(click, model, text):
    if text:
        tst = vectorizer.transform([text])
        if model == 'nb':
            return round(models[model]['obscene'].predict_proba(tst)[0][1], 3)
        else:
            return round(models[model]['obscene']._predict_proba_lr(tst)[0][1], 3)

@app.callback(
    Output("bar_chart", "figure"),
    [Input("check_button", "n_clicks"), Input('school_selection', 'value')],
    state=[State(component_id='input_text', component_property='value')])
def bar_update(click, model, text):
    if text:
        analyze_request = {
          'comment': { 'text': text },
          'requestedAttributes': {'TOXICITY': {}, 'SEVERE_TOXICITY': {}, 'THREAT': {}, 'OBSCENE': {}, 'IDENTITY_ATTACK': {}, 'INSULT': {}}
        }

        response = client.comments().analyze(body=analyze_request).execute()

        perpective = {val:response['attributeScores'][val]['summaryScore']['value'] for val in google_names}

        tst = vectorizer.transform([text])
        xvals = ['Toxic', 'Severe Toxic', 'Threat', 'Insult', 'Identity Hate', 'Obscene']
        yvals = []
        if model == 'nb':
            yvals.append(round(models[model]['toxic'].predict_proba(tst)[0][1], 3))
            yvals.append(round(models[model]['severe_toxic'].predict_proba(tst)[0][1], 3))
            yvals.append(round(models[model]['threat'].predict_proba(tst)[0][1], 3))
            yvals.append(round(models[model]['insult'].predict_proba(tst)[0][1], 3))
            yvals.append(round(models[model]['identity_hate'].predict_proba(tst)[0][1], 3))
            yvals.append(round(models[model]['obscene'].predict_proba(tst)[0][1], 3))
        else:
            yvals.append(round(models[model]['toxic']._predict_proba_lr(tst)[0][1], 3))
            yvals.append(round(models[model]['severe_toxic']._predict_proba_lr(tst)[0][1], 3))
            yvals.append(round(models[model]['threat']._predict_proba_lr(tst)[0][1], 3))
            yvals.append(round(models[model]['insult']._predict_proba_lr(tst)[0][1], 3))
            yvals.append(round(models[model]['identity_hate']._predict_proba_lr(tst)[0][1], 3))
            yvals.append(round(models[model]['obscene']._predict_proba_lr(tst)[0][1], 3))

        return {
        'data': [go.Bar(x = xvals, y = [i*100 for i in yvals], text = [round(i*100, 2) for i in yvals], textposition = 'auto', name = 'Toxic'),
                go.Bar(x = xvals, y = [perpective[i]*100 for i in google_names], text = [round(perpective[i]*100,2) for i in google_names], textposition = 'auto', name = 'Toxic_google')],
        'layout': go.Layout(
                            title = 'Toxicity',
                            yaxis = dict(title = 'Percentage of Toxicity', range = [0,100]),
                            height = 400,
                            )
            }
    else:
        return {
        'data': [go.Bar(x = ['Identity Hate', 'Threat', 'Severe Toxic', 'Insult', 'Obscene', 'Toxic'], y = [0,0,0,0,0,0], textposition = 'auto', name = 'Toxic'),
                go.Bar(x = ['Identity Hate', 'Threat', 'Severe Toxic', 'Insult', 'Obscene', 'Toxic'], y = [0,0,0,0,0,0], textposition = 'auto', name = 'Toxic_google')],
        'layout': go.Layout(
                            title = 'Toxicity',
                            yaxis = dict(title = 'Percentage of Toxicity', range = [0,100]),
                            height = 400,
                            )
            }


@app.callback(
    Output("roc_chart", "figure"),
    [Input('school_selection', 'value'), Input('color_selection', 'value')])
def roc_update(model, color):
    mod1 = model
    model = models[model][color]
    if mod1 == 'nb':
        y_score = model.predict_proba(testing_comments)[:, 1]
    else:
        y_score = model._predict_proba_lr(testing_comments)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test['toxic'], y_score)
    score = auc(fpr, tpr)

    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={score:.4f})',
        labels=dict(
            x='False Positive Rate',
            y='True Positive Rate'))
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1)

    fig.update_layout(
        title="ROC Curve",
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        # width=950, height=410
    )
    return fig

app.run_server(debug=True)
if __name__ == '__main__':
    app.run_server(debug=True)
