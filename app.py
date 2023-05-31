"""Methods to compute Michaelis-Menten equation parameters and statistics."""

from io import StringIO, BytesIO
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from bokeh.models.widgets.tables import NumberFormatter

import param

import panel as pn
from panel.widgets import (Button, CheckBoxGroup,
                           RadioButtonGroup, FileDownload)
pn.extension('tabulator', 'mathjax')
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
    """The Michaelis-Menten rate law."""
    return V * a / (Km + a)


class ResDict(param.Parameterized):
    """Class to hold data from a computation."""

    method = param.String(default='Hanes', doc='Name of the method used')
    error = param.String(default=None, doc='Computation error description')

    V = param.Number(default=0.0, bounds=(0.0, None),
                     doc='limiting rate')
    Km = param.Number(default=0.0, bounds=(0.0, None),
                      doc='Michaelis constant')

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
    intersections = param.Array(default=None,
                                doc='dlp intersection coordinates')
    dlp_lines = param.Array(default=None,
                            doc='slopes and intercepts of dlp lines')

# ------------ methods --------------------------
# all methods accept numpy arrays as input


def lineweaver_burk(a, v0):
    """Compute parameters by Lineweaver-Burk linearization."""
    x, y = 1/a, 1/v0
    m, b, Sm, Sb, _, _ = lin_regression(x, y)
    V = 1.0 / b
    Km = m / b
    SV = V * Sb / b
    SKm = Km * np.sqrt((Sm/m)**2 + (Sb/b)**2)
    return ResDict(method='Lineweaver-Burk',
                   V=V, Km=Km, SE_V=SV, SE_Km=SKm,
                   x=x, y=y, m=m, b=b)


def hanes_woolf(a, v0):
    """Compute parameters by Hanes linearization."""
    x = a
    y = a/v0
    m, b, Sm, Sb, _, _ = lin_regression(x, y)
    V = 1.0 / m
    Km = b / m
    SV = V * Sm / m
    SKm = Km * np.sqrt((Sm/m)**2 + (Sb/b)**2)
    return ResDict(method='Hanes',
                   V=V, Km=Km, SE_V=SV, SE_Km=SKm,
                   x=x, y=y, m=m, b=b)


def eadie_hofstee(a, v0):
    """Compute parameters by Eadie-Hofstee linearization."""
    x = v0/a
    y = v0
    m, b, Sm, Sb, _, _ = lin_regression(x, y)
    V = b
    Km = -m
    SV = Sb
    SKm = Sm
    return ResDict(method='Eadie-Hofstee',
                   V=V, Km=Km, SE_V=SV, SE_Km=SKm,
                   x=x, y=y, m=m, b=b)


def hyperbolic(a, v0):
    """Compute parameters by non-linear least-quares regression."""
    popt, pcov, *_ = curve_fit(MM, a, v0, p0=(max(v0), np.median(a)))
    errors = np.sqrt(np.diag(pcov))
    V, Km = popt[0:2]
    SV, SKm = errors[0:2]
    return ResDict(method='Hyperbolic regression',
                   V=V, Km=Km, SE_V=SV, SE_Km=SKm, x=a, y=v0)


def cornish_bowden(a, v0):
    """Compute parameters by the Direct Linear Plot."""
    straights = [(v/s, v) for v, s in zip(v0, a)]
    intersects = []

    for ((m1, b1), (m2, b2)) in combinations(straights, 2):
        xintersect = (b2 - b1) / (m1 - m2)
        yintersect = (b1 * m2 - b2 * m1) / (m2 - m1)
        intersects.append((xintersect, yintersect))
    intersects = np.array(intersects)

    Km, V = np.median(intersects, axis=0)
    # TODO: compute CIs

    # construct results
    try:
        res = ResDict(method='Eisenthal-C.Bowden',
                      V=V, Km=Km, x=a, y=v0,
                      intersections=intersects,
                      dlp_lines=np.array(straights))
    except ValueError as ve:
        res = ResDict(method='Eisenthal-C.Bowden',
                      error=f'Error while computing parameters by\nEisenthal-C.Bowden:\n{ve}')
    return res


def compute_methods(a, v0):
    """Compute results for all methods."""
    # remove points with zero
    nonzero = np.logical_and(a != 0, v0 != 0)
    a_nonzero = a.compress(nonzero)
    v0_nonzero = v0.compress(nonzero)

    # apply all methods
    m_table = (hyperbolic, lineweaver_burk,
               hanes_woolf, eadie_hofstee,
               cornish_bowden)
    results = [method(a_nonzero, v0_nonzero) for method in m_table]
    return {'a': a, 'v0': v0, 'results': results}

# ------------- (str) report of results ------


def repr_x_deltax(value, delta):
    if delta is None:
        return f'{value:6.3f}'
    return f"{value:6.3f} ± {delta:6.3f}"


def report_str(results):
    lines = []
    for result in results:
        lines.append(result.method)
        if result.error is not None:
            lines.append(str(result.error))
        else:
            lines.append('   V  = ' + repr_x_deltax(result.V, result.SE_V))
            lines.append('   Km = ' + repr_x_deltax(result.Km, result.SE_Km))
    return '\n'.join(lines)


# ------------ plots --------------------------

# constants

# demo data
# this is data from
# [Atkinson, M.R., Jackson, J.F., Morton, R.K.(1961) Biochem. J. 80(2):318-23]
# (https://doi.org/10.1042/bj0800318),
# used for method comparison by
# [Wilkinson, G.N. (1961) Biochem. J. 80(2):324–332]
# (https://doi.org/10.1042/bj0800324)

DEMO_DATA = """0.138 0.148
0.220 0.171
0.291 0.234
0.560 0.324
0.766 0.390
1.460 0.493
"""

default_color_scheme = ('darkviolet',
                        'tab:green',
                        'tab:red',
                        'tab:blue',
                        'tab:orange')

empty_df = pd.DataFrame({'substrate': [0], 'rate': [0]})
empty_df.index.name = '#'

all_methods_list = ('Hyperbolic Regression',
                    'Lineweaver-Burk',
                    'Hanes',
                    'Eadie-Hofstee',
                    'Eisenthal-C.Bowden')


def hypers_mpl(results=None,
               display_methods=all_methods_list,
               colorscheme=None,
               title=None,
               legend=True,
               grid=True):

    if results is None:
        fig0 = Figure(figsize=(5, 4), tight_layout=True)
        ax0 = fig0.subplots()
        ax0.text(0.5, 0.5, 'no figure generated')
        return fig0

    a = results['a']
    v0 = results['v0']
    all_results = results['results']
    plt.rc('mathtext', fontset='cm')
    f = Figure(figsize=(5, 4), tight_layout=True)
    ax = f.subplots()

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


def plot_others_mpl(results=None, colorscheme=None, grid=True):
    if results is None:
        fig1 = Figure(figsize=(9, 6))
        ax0 = fig1.subplots()
        ax0.text(0.5, 0.5, 'no figure generated')
        return fig1

    all_r = results['results']
    if colorscheme is None:
        colorscheme = default_color_scheme
    plt.rc('mathtext', fontset='cm')

    f = Figure(figsize=(9, 6))
    ax = f.subplots(2, 2)
    ax = ax.flatten()
    for i in range(0, 3):
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

    ax.plot([0, xmax], [result.b, ymax], color=color,
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

    if results.error is not None:
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.text(0.5, 0.5, 'no figure generated')
        return
    a = results.x
    intersections = results.intersections
    lines = results.dlp_lines
    if title is None:
        title = results.method


    # keep finite intersections (might be Inf)
    finite_intersections = np.logical_and(np.isfinite(intersections[:, 0]),
                                          np.isfinite(intersections[:, 1]))
    viz_intersections = np.compress(finite_intersections, intersections, axis=0)

    # print('intersections ---------------')
    # print(intersections)

    # print('viz_intersections ---------------')
    # print(viz_intersections)

    xmax = max(viz_intersections[:, 0]) * 1.1
    ymax = max(viz_intersections[:, 1]) * 1.1
    xmin = max(a) * 1.1

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
    for x, y in map(tuple, viz_intersections):
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


def read_data_df(data_text):
    tf = StringIO(data_text)
    df = pd.read_csv(tf, delim_whitespace=True, comment='#', index_col=False)
    a = df.iloc[:, 0].values()
    v0 = df.iloc[:, 1].values()
    return a, v0


# widgetry

# data input
data_input_text = pn.widgets.input.TextAreaInput(width=200,
                                                 sizing_mode='stretch_height',
                                                 min_height=200)

bokeh_formatters = {
    'rate': NumberFormatter(format='0.00000'),
    'substrate': NumberFormatter(format='0.00000'),
}
data_dfwidget = pn.widgets.Tabulator(empty_df, width=200, disabled=True,
                                     formatters=bokeh_formatters)

data_input_column = pn.Column(data_input_text, height=200)

# data input buttons
clear_button = Button(name='Clear', button_type='danger', width=80)


def b_clear(event):
    results_pane.visible = False
    data_input_text.value = data_input_text.placeholder
    edit_table_group.value = 'Edit'
    results_text.object = ''
    mpl_pane_hypers.object = hypers_mpl(results=None)
    mpl_pane_others.object = plot_others_mpl(results=None)


clear_button.on_click(b_clear)

edit_table_group = RadioButtonGroup(options=['Edit', 'Check'], width=100)
edit_table_group.value = 'Edit'

demo_button = Button(name='Demo data', width=200)


def b_demo(event):
    data_input_text.value = DEMO_DATA
    edit_table_group.value = 'Edit'


demo_button.on_click(b_demo)


def change_data_view(event):
    if event.new == 'Check':
        a, v0 = read_data(data_input_text.value)
        df = pd.DataFrame({'rate': v0}, index=a)
        df.index.name = 'substrate'

        data_dfwidget.value = df
        data_input_column[0] = data_dfwidget
        pn.state.notifications.position = 'top-left'
        pn.state.notifications.info('Data OK', duration=3000)
    else:
        data_input_column[0] = data_input_text


edit_table_group.param.watch(change_data_view, 'value')


# results

class MMResultsInterface(param.Parameterized):
    last_results = param.Dict(dict())
    check_methods = param.ListSelector(default=list(all_methods_list),
                                       objects=list(all_methods_list))

    # triggers new plots resulting from a new computation
    e = param.Event()

    @param.depends('check_methods', 'e', watch=True)
    def draw_main_plot(self):
        f = hypers_mpl(self.last_results,
                       display_methods=self.check_methods)
        mpl_pane_hypers.object = f

    @param.depends('e', watch=True)
    def draw_other_plots(self):
        f = plot_others_mpl(self.last_results)
        mpl_pane_others.object = f

    def get_png_hypers(self):
        if self.last_results is not None:
            f = hypers_mpl(self.last_results,
                           display_methods=self.check_methods)
            bio = BytesIO()
            f.savefig(bio, format='png', dpi=100)
            bio.seek(0)
            return bio
        return None

    def get_pdf_hypers(self):
        if self.last_results is not None:
            f = hypers_mpl(self.last_results,
                           display_methods=self.check_methods)
            bio = BytesIO()
            f.savefig(bio, format='pdf')
            bio.seek(0)
            return bio
        return None


results_handler = MMResultsInterface()

results_text = pn.pane.Str('', styles={'font-family': "monospace"})

fd_png = FileDownload(callback=results_handler.get_png_hypers,
                      filename='hypers.png', width=200)
fd_pdf = FileDownload(callback=results_handler.get_pdf_hypers,
                      filename='hypers.pdf', width=200)

# "Fit" button
fit_button = Button(name='Fit', width=200, button_type='primary')


def b_fit(event):
    # make results_pane visible and draw
    results_pane.visible = True

    # compute results
    s, v0 = read_data(data_input_text.value)
    results_handler.last_results = compute_methods(s, v0)

    # fill results text are and trigger the drawing of new plots
    results_text.object = report_str(results_handler.last_results['results'])
    results_handler.e = True


fit_button.on_click(b_fit)

top_buttons = pn.Row(edit_table_group, clear_button)

header = r"""## Michaelis-Menten equation fitting

$$v_o = \\frac{V a}{K_m + a}$$

by António Ferreira

### Data input
"""

# panes holding matplotlib plots
mpl_pane_hypers = pn.panel(hypers_mpl(results=None))
mpl_pane_others = pn.panel(plot_others_mpl(results=None))

data_input_row = pn.Row(data_input_column,
                        pn.Column(top_buttons, demo_button, fit_button))

method_choice = CheckBoxGroup.from_param(results_handler.param.check_methods,
                                         inline=False)

results_pane = pn.Column(pn.layout.Divider(), "### Parameter values",
                         results_text,
                         '### Plots',
                         pn.Row(mpl_pane_hypers,
                                pn.Column(pn.Spacer(height=50),
                                          method_choice,
                                          fd_png,
                                          fd_pdf)),
                         pn.Row(mpl_pane_others))

# start results pane hidden
results_pane.visible = False

app_column = pn.Column(pn.pane.Markdown(header, renderer='markdown'),
                       data_input_row,
                       results_pane)

app_column.servable(title='Michaelis-Menten fitting')
# app_column
