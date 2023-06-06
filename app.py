"""Methods to compute Michaelis-Menten equation parameters and statistics."""

from io import StringIO, BytesIO
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import (curve_fit, OptimizeWarning)
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from bokeh.models.widgets.tables import NumberFormatter, BooleanFormatter

import param

import panel as pn
from panel.widgets import (Button, CheckBoxGroup, FileDownload)
pn.extension('tabulator', 'mathjax')
pn.extension('notifications')
pn.extension(notifications=True)

# fitting methods section


class ResDict(param.Parameterized):
    """Class to hold data from a computation."""

    method = param.String(default='Hanes', doc='Name of the method used')
    error = param.String(default=None, doc='Computation error description')

    V = param.Number(default=0.0,  # bounds=(0.0, None),
                     doc='limiting rate')
    Km = param.Number(default=0.0,  # bounds=(0.0, None),
                      doc='Michaelis constant')

    # Optional, depending on the method:
    SE_V = param.Number(default=None,  # bounds=(0.0, None),
                        doc='standard error of the limiting rate')
    SE_Km = param.Number(default=None,  # bounds=(0.0, None),
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


error_msg = '''Error while computing parameters by
{}:
{}'''.format


def MM(a, V, Km):
    """The Michaelis-Menten rate law."""
    return V * a / (Km + a)


def lineweaver_burk(a, v0):
    """Compute parameters by Lineweaver-Burk linearization."""
    x, y = 1/a, 1/v0
    result = linregress(x, y)
    V = 1.0 / result.intercept
    Km = result.slope / result.intercept
    cv_m = result.stderr/result.slope
    cv_b = result.intercept_stderr/result.intercept
    SV = V * cv_b
    SKm = Km * np.sqrt(cv_m**2 + cv_b**2)
    res = ResDict(method='Lineweaver-Burk',
                  V=V, Km=Km, SE_V=SV, SE_Km=SKm,
                  x=x, y=y, m=result.slope, b=result.intercept)
    return res


def hanes_woolf(a, v0):
    """Compute parameters by Hanes linearization."""
    x, y = a, a/v0
    result = linregress(x, y)
    V = 1.0 / result.slope
    Km = result.intercept / result.slope
    cv_m = result.stderr/result.slope
    cv_b = result.intercept_stderr/result.intercept
    SV = V * cv_m
    SKm = Km * np.sqrt(cv_m**2 + cv_b**2)
    res = ResDict(method='Hanes',
                  V=V, Km=Km, SE_V=SV, SE_Km=SKm,
                  x=x, y=y, m=result.slope, b=result.intercept)
    return res


def eadie_hofstee(a, v0):
    """Compute parameters by Eadie-Hofstee linearization."""
    x, y = v0/a, v0
    result = linregress(x, y)
    V = result.intercept
    Km = -result.slope
    SV = result.intercept_stderr
    SKm = result.stderr
    res = ResDict(method='Eadie-Hofstee',
                  V=V, Km=Km, SE_V=SV, SE_Km=SKm,
                  x=x, y=y, m=result.slope, b=result.intercept)
    return res


def hyperbolic(a, v0):
    """Compute parameters by non-linear least-quares regression."""
    try:
        popt, pcov, *_ = curve_fit(MM, a, v0, p0=(max(v0), np.median(a)),
                                   full_output=True,
                                   check_finite=True)
        errors = np.sqrt(np.diag(pcov))
        V, Km = popt[0:2]
        SV, SKm = errors[0:2]
        res = ResDict(method='Hyperbolic regression',
                      V=V, Km=Km, SE_V=SV,
                      SE_Km=SKm, x=a, y=v0)
    except (ValueError, RuntimeError, OptimizeWarning) as error:
        res = ResDict(method='Hyperbolic regression',
                      error=error_msg('Hyperbolic regression', error),
                      V=0, Km=0, SE_V=0,
                      SE_Km=0, x=a, y=v0)
    return res


def cornish_bowden(a, v0):
    """Compute parameters by the Direct Linear Plot."""
    try:
        straights = [(v/s, v) for v, s in zip(v0, a)]
        intersects = []

        for ((m1, b1), (m2, b2)) in combinations(straights, 2):
            xintersect = (b2 - b1) / (m1 - m2)
            yintersect = (b1 * m2 - b2 * m1) / (m2 - m1)
            intersects.append((xintersect, yintersect))
        intersects = np.array(intersects)
        # print('intersects--------------')
        # print(intersects)

        Km, V = np.nanmedian(intersects, axis=0)
        # print('Km V--------------')
        # print(Km, V)
        # TODO: compute CIs

        # construct results
        res = ResDict(method='Eisenthal-C.Bowden',
                      V=V, Km=Km, x=a, y=v0,
                      intersections=intersects,
                      dlp_lines=np.array(straights))
    except ValueError as error:
        res = ResDict(method='Eisenthal-C.Bowden', x=a, y=v0,
                      error=error_msg('Eisenthal-C.Bowden', error))
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
            V = result.V
            if not np.isfinite(V) or V <= 0.0:
                lines.append(f'   Invalid V = {V}')
            else:
                lines.append('   V  = ' + repr_x_deltax(V, result.SE_V))
            Km = result.Km
            if not np.isfinite(Km) or Km <= 0.0:
                lines.append(f'   Invalid Km = {Km}')
            else:
                lines.append('   Km = ' + repr_x_deltax(Km, result.SE_Km))
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


all_methods_list = ('Hyperbolic Regression',
                    'Lineweaver-Burk',
                    'Hanes',
                    'Eadie-Hofstee',
                    'Eisenthal-C.Bowden')


def hypers_mpl(results=None, ax=None, plot_settings=None,
               title=None,
               legend=True,
               grid=True):

    if ax is None:
        return

    ax.clear()

    if results is None:
        ax.text(0.5, 0.5, 'no figure generated')
        return

    a = results['a']
    v0 = results['v0']
    all_results = results['results']

    plt.rc('mathtext', fontset='cm')
    # defaults
    colorscheme = default_color_scheme
    include_methods = all_methods_list
    # overide with plot_settings
    if plot_settings is not None:
        include_methods = plot_settings.include_methods

    xmax = max(a) * 1.1
    ymax = max(v0) * 1.1

    if title is not None:
        ax.set_title(title)
    ax.set_ylim(0, ymax)
    ax.set_xlim(0, xmax)
    chosen3letter = [choice[:3] for choice in include_methods]

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


def plot_others_mpl(results=None, f=None, colorscheme=None, grid=True):

    if f is None:
        return

    f.clear()

    if results is None:
        ax0 = f.subplots()
        ax0.text(0.5, 0.5, 'no figure generated')
        return

    all_r = results['results']

    if colorscheme is None:
        colorscheme = default_color_scheme
    plt.rc('mathtext', fontset='cm')

    ax = f.subplots(2, 2)
    ax = ax.flatten()
    # draw linearizations
    for i in range(0, 3):
        draw_lin_plot(ax[i], all_r[i+1],
                      color=colorscheme[i+1], grid=grid)
    # draw direct linear plot
    draw_cornish_bowden_plot(ax[3], all_r[4],
                             color=colorscheme[4], grid=grid)


def draw_lin_plot(ax, result, color='black',
                  title=None, grid=True):

    if title is None:
        title = result.method
    ax.set_title(title)

    if result.error is not None:
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.text(0.5, 0.5, 'no figure generated', ha='center')
        return
    x = result.x
    y = result.y

    xmax = max(x) * 1.1
    ymax = xmax * result.m + result.b

    if result.m < 0:
        ytop = result.b
    else:
        ytop = ymax
    ytop = 1.1 * ytop

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

    if title is None:
        title = results.method
    if results.error is not None:
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.text(0.5, 0.5, 'no figure generated', ha='center')
        return
    a = results.x
    intersections = results.intersections
    lines = results.dlp_lines

    # keep finite intersections (might be Inf)
    finite_intersections = np.logical_and(np.isfinite(intersections[:, 0]),
                                          np.isfinite(intersections[:, 1]))
    viz_intersections = np.compress(finite_intersections,
                                    intersections,
                                    axis=0)

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

    # plot straight lines through (-ai, 0) and (0, vi)
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


empty_df = pd.DataFrame({'substrate': [0.1],
                         'rate': [0.1],
                         'use': [True]})
empty_df.index.name = '#'

# data input
data_input_text = pn.widgets.input.TextAreaInput(width=200,
                                                 sizing_mode='stretch_height',
                                                 min_height=200)

bokeh_formatters = {
    'rate': NumberFormatter(format='0.00000'),
    'substrate': NumberFormatter(format='0.00000'),
    'use': BooleanFormatter()
}

data_dfwidget = pn.widgets.Tabulator(empty_df, width=300, show_index=False,
                                     selectable='checkbox',
                                     widths={'substrate': 100,
                                             'rate': 100},
                                     text_align={'use': 'center'},
                                     formatters=bokeh_formatters)

# data input buttons

add_row_button = Button(name='+', width=20)


def add_row(event):
    df = data_dfwidget.value
    new_row = pd.DataFrame({'substrate': [0.1], 'rate': [0.1], 'use': [True]})
    newdf = df.append(new_row, ignore_index=True)
    data_dfwidget.value = newdf


add_row_button.on_click(add_row)

# reset button
clear_button = Button(name='Reset', button_type='danger', width=150)


def b_clear(event):
    results_pane.visible = False
    data_input_text.value = data_input_text.placeholder
    data_dfwidget.value = empty_df
    edit_button.value = False
    # edit_table_group.value = 'Edit'
    results_text.object = ''

    hypers_mpl(results=None, ax=res_interface.f_ax['hypers_ax'])
    mpl_hypers.param.trigger('object')

    plot_others_mpl(results=None, f=res_interface.f_ax['others_f'])
    mpl_others.param.trigger('object')


clear_button.on_click(b_clear)

# Edit menu

menu_items = [('Copy', 'copy'),
              ('Paste', 'paste'),
              ('Cut', 'cut'),
              ('Invert selection', 'invert_selection'),
              None,
              ('Toggle "use"', 'toggle_use'),
              None,
              ('Delete unused', 'del_unused'),
              ('Delete selected', 'del_unused'),
              ('Clear all data', 'del_all')]

edit_button = pn.widgets.MenuButton(name='Edit', width=120,
                                    items=menu_items,
                                    button_type='success')


def handle_edit(event):
    choice = event.new
    if choice == 'del_all':
        data_dfwidget.value = empty_df
    elif choice == 'del_unused':
        df = data_dfwidget.value
        data_dfwidget.value = df[df.use]


edit_button.on_click(handle_edit)

# demo data
demo_button = Button(name='Demo')


def b_demo(event):
    # data_input_text.value = DEMO_DATA
    a, v0 = read_data(DEMO_DATA)
    demo_df = pd.DataFrame({'substrate': a, 'rate': v0, 'use': [True]*len(a)})
    # edit_table_group.value = 'Edit'
    data_dfwidget.value = demo_df


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


# results

class PlotSettings(param.Parameterized):
    include_methods = param.ListSelector(default=list(all_methods_list),
                                         objects=list(all_methods_list))


class MMResultsInterface(param.Parameterized):
    last_results = param.Dict({})

    plot_settings = PlotSettings()

    f_ax = param.Dict({})

    # triggers new plots resulting from V, Km computation
    e = param.Event()

    @param.depends('plot_settings.param', 'e', watch=True)
    def draw_main_plot(self):
        hypers_mpl(self.last_results, ax=self.f_ax['hypers_ax'],
                   plot_settings=self.plot_settings)
        mpl_hypers.param.trigger('object')

    @param.depends('e', watch=True)
    def draw_other_plots(self):
        plot_others_mpl(self.last_results, f=self.f_ax['others_f'])
        mpl_others.param.trigger('object')

    def get_png_hypers(self):
        if self.last_results is not None:
            hypers_mpl(self.last_results, ax=self.f_ax['hypers_ax'],
                       plot_settings=self.plot_settings)
            f = self.f_ax['hypers_f']
            bio = BytesIO()
            f.savefig(bio, format='png', dpi=100)
            bio.seek(0)
            return bio
        return None

    def get_pdf_hypers(self):
        if self.last_results is not None:
            hypers_mpl(self.last_results, ax=self.f_ax['hypers_ax'],
                       plot_settings=self.plot_settings)
            f = self.f_ax['hypers_f']
            bio = BytesIO()
            f.savefig(bio, format='pdf')
            bio.seek(0)
            return bio
        return None


res_interface = MMResultsInterface()

results_text = pn.pane.Str('', styles={'font-family': "monospace",
                                       'font-size': '12pt'})

# "Fit" button
fit_button = Button(name='Fit', width=150,
                    button_type='primary',
                    icon='calculator', icon_size='2em')


def b_fit(event):
    # make results_pane visible and draw
    results_pane.visible = True

    # compute results
    # s, v0 = read_data(data_input_text.value)
    df = data_dfwidget.value
    df = df[df.use]
    # TODO: validate
    subs_conc = df['substrate'].values
    v0_values = df['rate'].values
    res_interface.last_results = compute_methods(subs_conc, v0_values)

    # fill results text and trigger the drawing of new plots
    results_text.object = report_str(res_interface.last_results['results'])
    res_interface.e = True


fit_button.on_click(b_fit)

edit_buttons = pn.Row(add_row_button,
                      demo_button,
                      edit_button)

data_input_column = pn.Column(edit_buttons, data_dfwidget, height=250)
# data_input_column = pn.Column(data_input_text, height=200)

header = pn.pane.Markdown(r"""## Michaelis-Menten equation fitting

$$v_o = \\frac{V a}{K_m + a}$$

by António Ferreira

### Data input
""", renderer='markdown')

# figures holding matplotlib plots


def init_figures():
    # setup hypers figure
    f = Figure(figsize=(5, 4), tight_layout=True)
    ax = f.subplots()
    res_interface.f_ax['hypers_f'] = f
    res_interface.f_ax['hypers_ax'] = ax
    hypers_mpl(results=None, ax=ax)

    # setup "other plots" figure
    f = Figure(figsize=(9, 6), tight_layout=True)
    res_interface.f_ax['others_f'] = f
    plot_others_mpl(results=None, f=f)


init_figures()
mpl_hypers = pn.pane.Matplotlib(res_interface.f_ax['hypers_f'])
mpl_others = pn.pane.Matplotlib(res_interface.f_ax['others_f'])

data_input_row = pn.Row(pn.WidgetBox(data_input_column, height=320),
                        pn.Column(fit_button, clear_button))

# plot settings
method_choice = CheckBoxGroup.from_param(res_interface.
                                         plot_settings.param.
                                         include_methods,
                                         inline=False)

download_png = FileDownload(callback=res_interface.get_png_hypers,
                            filename='hypers.png', width=200)
download_pdf = FileDownload(callback=res_interface.get_pdf_hypers,
                            filename='hypers.pdf', width=200)

plot_settings = pn.Column("#### Include",
                          method_choice,
                          download_png,
                          download_pdf)

# plots
tabs = pn.Tabs(('MM equation plot', mpl_hypers),
               ('Secondary plots', mpl_others))
plots_box = pn.WidgetBox(tabs)

results_pane = pn.Column(pn.layout.Divider(), "### Parameter values",
                         results_text,
                         pn.Spacer(height=50),
                         pn.Row(plot_settings, plots_box))

# start results pane hidden
results_pane.visible = False

app_column = pn.Column(header, data_input_row, results_pane)

app_column.servable(title='Michaelis-Menten fitting')
# app_column
