"""Methods to compute Michaelis-Menten equation parameters and statistics.

   Wilkinson [TODO: insert reference] data is used to test the methods."""

from io import StringIO, BytesIO
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from bokeh.models.widgets.tables import NumberFormatter, BooleanFormatter

import param
from param import Parameter, Parameterized
import panel as pn
pn.extension('tabulator')
pn.extension('notifications')
pn.extension(notifications=True)

# fitting methods section

def lin_regression(xvalues, yvalues):
    """Simple linear regression (y = m * x + b + error)."""
    m, b, R, p, SEm = linregress(xvalues, yvalues)

    # need to compute SEb, linregress only computes SEm
    n = len(xvalues)
    SSx = np.var(xvalues, ddof=1) * (n-1)  # this is sum( (x - mean(x))**2 )
    SEb2 = SEm**2 * (SSx/n + np.mean(xvalues)**2)
    SEb = SEb2**0.5

    return m, b, SEm, SEb, R, p


def MM(a, V, Km):
    return V * a / (Km + a)


class ResDict(Parameterized):
    """Class to hold data from a computation."""

    method = param.String(default='Hanes', doc='Name of the method used')
    error = param.String(default=None, doc='Computation error description')

    V = param.Number(default=0.0, bounds=(0.0, None), doc='limiting rate')
    Km = param.Number(default=0.0, bounds=(0.0, None), doc='Michaelis constant')

    # Optional, depending on the method:
    SE_V = param.Number(default=None, bounds=(0.0, None),
                       doc='standard error of the limiting rate')
    SE_Km = param.Number(default=None, bounds=(0.0, None),
                       doc='standard error of the Michelis constant')

    # Optional for linearizations:
    m = param.Number(default=None, doc='slope of linearization')
    b = param.Number(default=None, doc='intercept of linearization')
    x = param.Array(default=None, doc='x-values during linearization')
    y = param.Array(default=None, doc='y-values during linearization')

    # Optional for direct liner plot:
    intersections = param.Array(default=None, doc='dlp intersection coordinates')
    dlp_lines = param.Array(default=None, doc='slopes and intercepts of dlp lines')

# ------------ methods --------------------------
# all methods accept numpy arrays as input

def lineweaver_burk(a, v0):
    while 0 in a:
        index = np.where(a == 0)
        a = np.delete(a, index)
        v0 = np.delete(v0, index)
    while 0 in v0:
        index = np.where(v0 == 0)
        a = np.delete(a, index)
        v0 = np.delete(v0, index)
    x, y = 1/a, 1/v0
    m, b, Sm, Sb, _, _ = lin_regression(x, y)
    V = 1.0 / b
    Km = m / b
    SV = V * Sb / b
    SKm = Km * np.sqrt((Sm/m)**2 + (Sb/b)**2)
    return ResDict(method='Lineweaver-Burk', V=V, Km=Km, SE_V=SV, SE_Km=SKm,
                    x=x, y=y, m=m, b=b)


def hanes_woolf(a, v0):
    while 0 in a:
        index = np.where(a == 0)
        a = np.delete(a, index)
        v0 = np.delete(v0, index)
    while 0 in v0:
        index = np.where(v0 == 0)
        a = np.delete(a, index)
        v0 = np.delete(v0, index)
    x = a
    y = a/v0
    m, b, Sm, Sb, _, _ = lin_regression(x, y)
    V = 1.0 / m
    Km = b / m
    SV = V * Sm / m
    SKm = Km * np.sqrt((Sm/m)**2 + (Sb/b)**2)
    return ResDict(method='Hanes', V=V, Km=Km, SE_V=SV, SE_Km=SKm,
                    x=x, y=y, m=m, b=b)


def eadie_hofstee(a, v0):
    while 0 in a:
        index = np.where(a == 0)
        a = np.delete(a, index)
        v0 = np.delete(v0, index)
    while 0 in v0:
        index = np.where(v0 == 0)
        a = np.delete(a, index)
        v0 = np.delete(v0, index)
    x = v0/a
    y = v0
    m, b, Sm, Sb, _, _ = lin_regression(x, y)
    V = b
    Km = -m
    SV = Sb
    SKm = Sm
    return ResDict(method='Eadie-Hofstee', V=V, Km=Km, SE_V=SV, SE_Km=SKm,
                    x=x, y=y, m=m, b=b)


def hyperbolic(a, v0):
    popt, pcov, *_ = curve_fit(MM, a, v0, p0=(max(v0), np.median(a)))
    errors = np.sqrt(np.diag(pcov))
    V, Km = popt[0:2]
    SV, SKm = errors[0:2]
    return ResDict(method='Hyperbolic regression', V=V, Km=Km, SE_V=SV, SE_Km=SKm, x=a, y=v0)


def cornish_bowden(a, v0):
    straights = [(v/s, v) for v, s in zip(v0, a)]
    intersects = []

    for ((m1, b1), (m2, b2)) in combinations(straights, 2):
        xintersect = (b2 - b1) / (m1 - m2)
        yintersect = (b1 * m2 - b2 * m1) / (m2 -m1)
        intersects.append((xintersect, yintersect))
    intersects = np.array(intersects)

    Km, V = np.median(intersects, axis=0)
    # TODO: compute CIs
    return ResDict(method='Eisenthal-C.Bowden',
                   V=V, Km=Km, x=a, y=v0,
                   intersections=intersects,
                   dlp_lines=np.array(straights))


def compute_methods(a, v0):
    """Compute results for all methods."""
    m_table  = (hyperbolic, lineweaver_burk, hanes_woolf, eadie_hofstee, cornish_bowden)
    results = [m(a, v0) for m in m_table]
    return {'a': a, 'v0': v0, 'results': results}

# ------------- (str) report of results ------

def repr_x_deltax(x, deltax):
    if deltax is None:
        return f'{x:6.3f}'
    return f"{x:6.3f} ± {deltax:6.3f}"

def report_str(results):
    lines = []
    for result in results:
        lines.append(result.method)
        lines.append('   V  = ' + repr_x_deltax(result.V, result.SE_V))
        lines.append('   Km = ' + repr_x_deltax(result.Km, result.SE_Km))
    return '\n'.join(lines)

# ------------ plots --------------------------

# constants
demo_data = """0.138 0.148
0.220 0.171
0.291 0.234
0.560 0.324
0.766 0.390
1.460 0.493
"""

default_color_scheme = ('darkviolet',
                        'green',
                        'darkred',
                        'cornflowerblue',
                        'goldenrod')

empty_df = pd.DataFrame({'substrate': [0], 'rate':[0]})
empty_df.index.name = '#'

fig0 = Figure(figsize=(8, 6))
ax0 = fig0.subplots()
t = ax0.text(0.5, 0.5, 'no figure generated')

fig1 = Figure(figsize=(12, 8))
ax0 = fig1.subplots()
t = ax0.text(0.5, 0.5, 'no figure generated')

last_results = None
all_methods_list = ['Hyperbolic Regression',
                    'Lineweaver-Burk',
                    'Hanes',
                    'Eadie-Hofstee',
                    'Eisenthal-C.Bowden']

def read_data(data):
    a = []
    v0 = []
    for line in data.splitlines():
        line = line.strip()
        if len(line) == 0:
            continue
        if line.startswith('#'):
            continue
        x1, x2 = line.split(None, 2)
        try:
            x1, x2 = float(x1), float(x2)
        except ValueError:
            continue
        a.append(x1)
        v0.append(x2)
    return np.array(a), np.array(v0)


def hypers_mpl(results,
               display_methods=tuple(all_methods_list),
               colorscheme=None,
               title=None,
               legend=True,
               grid=True):

    a = results['a']
    v0 = results['v0']
    all_results = results['results']
    plt.rc('mathtext', fontset='cm')
    f, ax = plt.subplots(1, 1, figsize=(8,6))

    if colorscheme is None:
        colorscheme = default_color_scheme

    xmax = max(a) * 1.1
    ymax = max(v0) * 1.1

    if title is not None:
        ax.set_title(title)
    ax.set_ylim(0, ymax)
    ax.set_xlim(0, xmax)
    chosen3letter = [choice[:3] for choice in display_methods]

    for result, color in zip(all_results, colorscheme):
        if result.method[:3] not in chosen3letter:
            continue

        V, Km = result.V, result.Km
        line_x = np.linspace(0.0, xmax, 200)
        line_y = MM(line_x, V, Km)

        ax.plot(line_x, line_y, label=result.method,
                color=color, linestyle='solid', lw=2)

    ax.plot(a, v0, marker='o',
                   linestyle='None',
                   markerfacecolor='white',
                   markeredgecolor='black',
                   markeredgewidth=1.5,
                   markersize=6)
    ax.set_xlabel('$a$', fontsize=16)
    ax.set_ylabel('$v_o$', fontsize=16)
    if legend:
        ax.legend(loc='lower right')
    if grid:
        ax.grid()
    return f


def plot_others_mpl(results, colorscheme=None, grid=True):
    all_r = results['results']
    if colorscheme is None:
        colorscheme = default_color_scheme
    plt.rc('mathtext', fontset='cm')
    f, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax = [ax[0][0], ax[0][1], ax[1][0], ax[1][1]]
    for i in range(0,3):
        draw_lin_plot(ax[i], all_r[i+1], color=colorscheme[i+1], grid=grid)
    draw_cornish_bowden_plot(ax[3], all_r[4], color=colorscheme[4], grid=grid)
    f.tight_layout()
    return f


def draw_lin_plot(ax, result, color='black',
                  title=None, grid=True):

    if title is None:
        title = result.method
    x = result.x
    y = result.y

    xmax = max(x) * 1.1
    ymax = xmax * result.m + result.b

    if result.m < 0:
        ytop = result.b
    else:
        ytop = ymax
    ytop = 1.1 * ytop

    ax.set_title(title)
    ax.set_ylim(0, ytop)
    ax.set_xlim(0, xmax)

    ax.plot([0,xmax], [result.b, ymax], color=color,
                  linestyle='solid', lw=2)

    ax.plot(x, y,  marker='o',
                   linestyle='None',
                   markerfacecolor='white',
                   markeredgecolor='black',
                   markeredgewidth=1.5,
                   markersize=6)

    if result.method.startswith('Lineweaver'):
        xlabel = '$1/a$'
        ylabel = '$1/v_o$'
    elif result.method.startswith('Hanes'):
        xlabel = '$a$'
        ylabel = '$a/v_o$'
    elif result.method.startswith('Eadie'):
        xlabel = '$v_o/a$'
        ylabel = '$v_o$'
    else:
        xlabel = ''
        ylabel = ''
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if grid:
        ax.grid()

def draw_cornish_bowden_plot(ax, results,
                             color='black',
                             title=None,
                             grid=True):

    a = results.x
    v0 = results.y
    intersections = results.intersections
    lines = results.dlp_lines
    if title is None:
        title = results.method

    xmax = max(intersections[:, 0]) * 1.1
    ymax = max(intersections[:, 1]) * 1.1
    xmin = max(a) * 1.1
    ymin = 0.0

    ax.set_title(title)
    ax.set_ylim(0, ymax)
    ax.set_xlim(-xmin, xmax)

    # plot lines
    for m, b in lines:
        ymaxi = m * xmax + b
        ax.plot([-b/m, xmax], [0, ymaxi],
                  color='gray',
                  linestyle='solid',
                  lw=1)

    # plot intersection points
    for x, y in map(tuple, intersections):
        ax.plot(x, y,
                   marker='o',
                   linestyle='None',
                   markerfacecolor='white',
                   markeredgecolor='black',
                   markeredgewidth=1,
                   markersize=4)

    # plot median intersection
    ax.plot(results.Km, results.V,
               marker='o',
               linestyle='None',
               markerfacecolor='white',
               markeredgecolor=color,
               markeredgewidth=1.5,
               markersize=6)
    ax.set_xlabel('Km')
    ax.set_ylabel('V')
    if grid:
        ax.grid()

def read_data_df(data_text):
    tf = StringIO(data_text)
    df = pd.read_csv(tf, delim_whitespace=True, comment='#', index_col=False)
    a = df.iloc[:, 0].values()
    v0 = df.iloc[:, 1].values()
    return a, v0


# widgetry

# data input
data_input_text = pn.widgets.input.TextAreaInput(height=300, width=200,
                                                 min_height=300,
                                                 min_width=100,
                                                 height_policy='min')

#data_dfwidget = pn.widgets.DataFrame(empty_df, width=300, disabled=True)
bokeh_formatters = {
    'rate': NumberFormatter(format='0.00000'),
    'substrate': NumberFormatter(format='0.00000'),
}
data_dfwidget = pn.widgets.Tabulator(empty_df, width=200, disabled=True,
                                     formatters=bokeh_formatters)

data_input_column = pn.Column(data_input_text)

# data input buttons
clear_button = pn.widgets.Button(name='Clear', button_type='danger', width=80)
def b_clear(event):
    results_pane.visible = False
    data_input_text.value = data_input_text.placeholder
    edit_table_group.value = 'Edit'
    results_text.object = ''
    ready_check.value = False
clear_button.on_click(b_clear)

edit_table_group = pn.widgets.RadioButtonGroup(options=['Edit', 'Check'], width=100)
edit_table_group.value = 'Edit'

demo_button = pn.widgets.Button(name='Demo data', width=200)
def b_demo(event):
    data_input_text.value = demo_data
    edit_table_group.value = 'Edit'
demo_button.on_click(b_demo)

def change_data_view(event):
    if event.new == 'Check':
        a, v0 = read_data(data_input_text.value)
        df = pd.DataFrame({'rate':v0}, index=a)
        df.index.name = 'substrate'

        data_dfwidget.value = df
        data_input_column[0] = data_dfwidget
        pn.state.notifications.position = 'top-left'
        pn.state.notifications.info('Data OK', duration=3000)
    else:
        data_input_column[0] = data_input_text

edit_table_group.param.watch(change_data_view, 'value')

# results

results_text = pn.pane.Str('', style={'font-family': "monospace"})

check_methods = pn.widgets.CheckBoxGroup(options=all_methods_list,
                                         value=all_methods_list,
                                         inline=False)

ready_check = pn.widgets.Checkbox(name='Ready')
ready_check.value = False
ready_check.visible = False

@pn.depends(check_methods, ready_check)
def draw_main_plot(check_methods, ready_check):
    global last_results
    if ready_check:
        f = hypers_mpl(last_results, display_methods=check_methods)
    else:
        f = fig0
    return pn.pane.Matplotlib(f)

@pn.depends(check_methods, ready_check)
def draw_other_plots(check_methods, ready_check):
    global last_results
    if ready_check:
        f = plot_others_mpl(last_results)
    else:
        f = fig1
    return pn.pane.Matplotlib(f)

@pn.depends(check_methods)
def get_png_hypers(check_methods):
    global last_results
    if last_results is not None:
        f = hypers_mpl(last_results, display_methods=check_methods)
        bio = BytesIO()
        f.savefig(bio, format='png', dpi=100)
        bio.seek(0)
        return bio
    return None

@pn.depends(check_methods)
def get_pdf_hypers(check_methods):
    global last_results
    if last_results is not None:
        f = hypers_mpl(last_results, display_methods=check_methods)
        bio = BytesIO()
        f.savefig(bio, format='pdf')
        bio.seek(0)
        return bio
    return None

fd_png = pn.widgets.FileDownload(callback=get_png_hypers, filename='hypers.png', width=200)
fd_pdf = pn.widgets.FileDownload(callback=get_pdf_hypers, filename='hypers.pdf', width=200)

# "Fit" button
fit_button = pn.widgets.Button(name='Fit', width=200, button_type='primary')
def b_fit(event):
    global last_results
    results_pane.visible = True
    ready_check.value = False
    s, v0 = read_data(data_input_text.value)
    last_results = compute_methods(s, v0)

    results_text.object = report_str(last_results['results'])
    ready_check.value = True

fit_button.on_click(b_fit)

top_buttons = pn.Row(edit_table_group, clear_button)

header = """## Michaelis-Menten equation fitting

$$v_o = \\frac{V a}{K_m + a}$$

by António Ferreira

### Data input
"""

data_input_row = pn.Row(data_input_column, pn.Column(top_buttons, demo_button, fit_button))

results_pane = pn.Column(pn.layout.Divider(), "### Parameter values",
                         results_text,
                         ready_check, # remains hidden
                         '### Plots',
                         pn.Row(draw_main_plot,
                            pn.Column(pn.Spacer(height=50),
                                        check_methods,
                                        fd_png,
                                        fd_pdf)),
                         pn.Row(draw_other_plots))
results_pane.visible = False

app_column = pn.Column(header, data_input_row, results_pane)

app_column.servable()
# app_column
