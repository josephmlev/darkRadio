import plotly.graph_objects as go; import numpy as np
from plotly_resampler import register_plotly_resampler
import trace_updater
import panel as pn
import numpy as np
import dash
from dash import Dash, dcc, html
from base64 import b64encode
import io
import time

while True:
    app = Dash(__name__)
    buffer = io.StringIO()

    # Call the register function once and all Figures/FigureWidgets will be wrapped
    # according to the register_plotly_resampler its `mode` argument
    register_plotly_resampler(mode='MinMaxAggreator')

    test = np.loadtxt('asdf.txt', dtype=str, delimiter=',')

    numPoints = 2**23

    # generate some data
    x = np.linspace(0, numPoints, numPoints)
    noise = np.random.normal(0, 1, numPoints)
    undulation = 2*np.sin(x * 4*2*np.pi/numPoints) + 2/numPoints*x

    data = noise+undulation
    data [3500000] = 6 + undulation[3500000]

    print(data[3000000])
    print(undulation[3000000])


    # when working in an IPython environment, this will automatically be a
    # FigureWidgetResampler else, this will be an FigureResampler
    fig = go.Figure()
    fig.add_trace({"y": data, "name": "yp2"},)
    fig.add_trace({"y": undulation})

    fig.write_html(buffer)


    app.layout = html.Div([
        dcc.Graph(id="graph-id", figure=fig),
        trace_updater.TraceUpdater(id="trace-updater", gdID="graph-id"),
        dcc.Interval(id='interval-component', interval=1000, n_intervals=0),
    ])
    print('here')
    # Register the callback
    fig.register_update_graph_callback(app, "graph-id", "trace-updater")

    app.run_server(debug=True)

    time.sleep(1)