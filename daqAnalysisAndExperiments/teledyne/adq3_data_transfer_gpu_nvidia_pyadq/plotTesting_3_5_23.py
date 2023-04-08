import plotly.graph_objects as go; import numpy as np
from plotly_resampler import register_plotly_resampler, FigureResampler
import trace_updater
import panel as pn
import numpy as np
import dash
from plotly.subplots import make_subplots
from dash import Dash, dcc, html
from base64 import b64encode
import io
import time
from dash.dependencies import Input, Output
import settings as s
import matplotlib.pyplot as plt
import os


print("If you get ValueError: Cannot reshape array... Just resart code!")
print("If it reads a file while it is being written this may happen")
print("It is safe after the code is initilized")

app = Dash(__name__)
buffer = io.StringIO()

register_plotly_resampler(mode='MinMaxAggreator', default_n_shown_samples=10000)


fig = go.Figure()
fig = make_subplots(
    rows=4, cols=1,
    subplot_titles=(
        "Main Channel (Instantaneous)",
        "Veto Channel (Instantaneous)",
        "Main Channel (Average)",
        "Veto Channel (Average)"
    ),
    shared_xaxes=True,
    vertical_spacing=0.05,
    x_title='Frequency (MHz)',
    y_title='Power (dBm)'
)

fig.update_layout(height=2000, width=2000,
    title_text="Dark Radio Heads Up <br><sup></sup>",
    font=dict(
        family="Courier New, monospace",
        size=22,
        color="Black",
    ),
    margin=dict(t=250)
)


fig.update_annotations(font_size=22)


chA_switch0 = 10*np.log10(np.load(s.SAVE_DIRECTORY + 'plottingSpec/chA_W_switch0.npy')*1000)
chB_switch0 = 10*np.log10(np.load(s.SAVE_DIRECTORY + 'plottingSpec/chB_W_switch0.npy')*1000)
if s.SWITCH:
    chA_switch1 = 10*np.log10(np.load(s.SAVE_DIRECTORY + 'plottingSpec/chA_W_switch1.npy')*1000)
    chB_switch1 = 10*np.log10(np.load(s.SAVE_DIRECTORY + 'plottingSpec/chB_W_switch1.npy')*1000)

fig.add_trace(
    {"y": chB_switch0, "name": "antSpecInst"},
    row = 1, col = 1,
)
if s.SWITCH:
    fig.add_trace(
        {"y": chB_switch1, "name": "termSpecInst"},
        row = 1, col = 1,
    )
fig.add_trace(
    {"y": chA_switch0, "name": "termSpecInst"},
    row = 2, col = 1,
)

fig.write_html(buffer)


app.layout = html.Div([
    dcc.Graph(id="graph-id", figure=fig),
    trace_updater.TraceUpdater(id="trace-updater", gdID="graph-id"),
    dcc.Interval(id='interval-component', interval=99999999999, n_intervals=0),
])

# Register the callback
fig.register_update_graph_callback(app, "graph-id", "trace-updater")

# Multiple components can update everytime interval gets fired.
@app.callback(Output('graph-id', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n):
    print('data loop')
    #fig.register_update_graph_callback(app, "graph-id", "trace-updater")
    try:

        if not os.path.exists(s.SAVE_DIRECTORY+'database.txt'):
            acqNum = 0
            print(f'set to 0')
        else:
            numAcqs = int(open(s.SAVE_DIRECTORY + 'database.txt', 'r'
                            ).readlines()[-1].split(',')[0].strip()) + 1 
            datetimei   = str(open(s.SAVE_DIRECTORY + 'database.txt', 'r'
                            ).readlines()[-1].split(',')[1].strip())
            datetimef   = str(open(s.SAVE_DIRECTORY + 'database.txt', 'r'
                            ).readlines()[1].split(',')[1].strip())

        fig.update_layout(height=1300, width=1500,
            title_text=(f"Dark Radio Heads Up" +
                f"<br><sup>{numAcqs*s.NOF_BUFFERS_TO_RECEIVE} Averages at {datetimei}</sup>"+
                f"<br><sup>Run Started at    {datetimef}</sup>"
            ),
            font=dict(
            family="Courier New, monospace",
            size=26,
            color="Black",
        )
)

        freqs       = np.linspace(0, s.SAMPLE_RATE/2 / 1000000, s.CH0_RECORD_LEN//2)

        chA_switch0 = 10*np.log10(np.load(s.SAVE_DIRECTORY + 'plottingSpec/chA_W_switch0.npy')*1000)[1:]
        chB_switch0 = 10*np.log10(np.load(s.SAVE_DIRECTORY + 'plottingSpec/chB_W_switch0.npy')*1000)[1:]
        if s.SWITCH:
            #chA_switch1 = 10*np.log10(np.load(s.SAVE_DIRECTORY + 'plottingSpec/chA_W_switch1.npy')*1000)[1:]
            chB_switch1 = 10*np.log10(np.load(s.SAVE_DIRECTORY + 'plottingSpec/chB_W_switch1.npy')*1000)[1:]
        
        chA_avg_switch0 = 10*np.log10(np.load(s.SAVE_DIRECTORY + 'plottingSpec/chA_avg_W_switch0.npy')*1000)[1:]
        chB_avg_switch0 = 10*np.log10(np.load(s.SAVE_DIRECTORY + 'plottingSpec/chB_avg_W_switch0.npy')*1000)[1:]
        if s.SWITCH:
            #chA_avg_switch1 = 10*np.log10(np.load(s.SAVE_DIRECTORY + 'plottingSpec/chA_avg_W_switch1.npy')*1000)[1:]
            chB_avg_switch1 = 10*np.log10(np.load(s.SAVE_DIRECTORY + 'plottingSpec/chB_avg_W_switch1.npy')*1000)[1:]

        fig.data = []
        
        ########Plot 1: Main Ch, Instantaneous########
        fig.add_trace(
            {"x": freqs[1000:], "y": chB_switch0[1000:], "name": "Antenna Instantaneous"},
            row = 1, col = 1,
        )
        if s.SWITCH:
            fig.add_trace(
                {"x":freqs[1000:], "y": chB_switch1[1000:], "name": "Terminator Instantaneous"},
                row = 1, col = 1,
            )
        ########Plot 2: Veto Ch, Instantaneous########
        fig.add_trace(
            {"x": freqs[1000:], "y": chA_switch0[1000:], "name": "Veto Instantaneous"},
            row = 2, col = 1,
        )
        
        ########Plot 3: Main Ch, Average########
        fig.add_trace(
            {"x": freqs[1000:], "y": chB_avg_switch0[1000:], "name": "Antenna Average"},
            row = 3, col = 1,
        )
        if s.SWITCH:
            fig.add_trace(
                {"x":freqs[1000:], "y": chB_avg_switch1[1000:], "name": "Terminator Average"},
                row = 3, col = 1,
            )
        ########Plot 4: Veto Ch, Average########
        fig.add_trace(
            {"x": freqs[1000:], "y": chA_avg_switch0[1000:], "name": "Veto Average"},
            row = 4, col = 1,
        )
        
    except:
        print('Error encountered in plotting code. Likley reading data before it was written')
        pass
    return fig
app.run_server(debug=True)
