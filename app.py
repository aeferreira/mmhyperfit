"""Methods to compute Michaelis-Menten equation parameters and statistics.

   Wilkinson [TODO: insert reference] data is used to test the methods."""

from io import StringIO, BytesIO

#from methods import compute_methods, MM_line
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress as lreg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas

import panel as pn
pn.extension('tabulator')

# fitting methods section

def lin_regression(x, y):
    """Simple linear regression (y = m * x + b + error)."""
    m, b, R, p, SEm = lreg(x, y)

    # need to compute SEb, linregress only computes SEm
    n = len(x)
    SSx = np.var(x, ddof=1) * (n-1)  # this is sum( (x - mean(x))**2 )
    SEb2 = SEm**2 * (SSx/n + np.mean(x)**2)
    SEb = SEb2**0.5

    return m, b, SEm, SEb, R, p


def MM(a, V, Km):
    return V * a / (Km + a)


def MM_line(V, Km, xmax=1.0):
    x0 = 0
    x = np.linspace(x0, xmax, 200)
    return x, MM(x, V, Km)


def res_tuple(method, V, Km, SV=None, SKm=None):
    sV = f"{V:6.4f}"
    sKm = f"{Km:6.4f}"
    if SV is None:
        sSV = 'n/a'
    else:
        sSV = f"{SV:6.4f}"
    if SKm is None:
        sSKm = 'n/a'
    else:
        sSKm = f"{SKm:6.4f}"
    return method, sV, sSV, sKm, sSKm



# dict that holds data from a computation.

# Mandatory members are:
# name - Name of the method used (str).
# error - None by default, str describing error in computation
# V - limiting rate (float)
# Km - Michaelis constant (float)

# Optional, depending on the method:
# SE_V - standard error of the limiting rate
# SE_Km - standard error of the Michelis constant

# Optional for linearizations:
# x - x-values during linearization (iterable of floats)
# y - y-values during linearization (iterable of floats)
# m - slope of linearization
# b - intercept of linearization


def res_object(method, V=0.0, Km=0.0, SE_V=None, SE_Km=None, error=None,
               x=None, y=None, m=None, b=None):
    return {'name': method, 'V': V, 'Km':Km, 'error': error,
            'SE_V': SE_V, 'SE_Km': SE_Km,
            'x':x, 'y':y, 'm':m, 'b':b}


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
    m, b, Sm, Sb, R, p = lin_regression(x, y)
    V = 1.0 / b
    Km = m / b
    SV = V * Sb / b
    SKm = Km * np.sqrt((Sm/m)**2 + (Sb/b)**2)
    return res_object('Lineweaver-Burk', V, Km, SE_V=SV, SE_Km=SKm,
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
    m, b, Sm, Sb, R, p = lin_regression(x, y)
    V = 1.0 / m
    Km = b / m
    SV = V * Sm / m
    SKm = Km * np.sqrt((Sm/m)**2 + (Sb/b)**2)
    return res_object('Hanes or Woolf', V, Km, SE_V=SV, SE_Km=SKm,
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
    m, b, Sm, Sb, R, p = lin_regression(x, y)
    V = b
    Km = -m
    SV = Sb
    SKm = Sm
    return res_object('Eadie-Hofstee', V, Km, SE_V=SV, SE_Km=SKm,
                      x=x, y=y, m=m, b=b)


def hyperbolic(a, v0):
    popt, pcov = curve_fit(MM, a, v0, p0=(max(v0), np.median(a)))
    errors = np.sqrt(np.diag(pcov))
    V, Km = popt[0:2]
    SV, SKm = errors[0:2]
    return res_object('Hyperbolic', V, Km, SE_V=SV, SE_Km=SKm, x=a, y=v0)


def cornish_bowden(a, v0):
    straights = [(v/s, v) for v, s in zip(v0, a)]
    intersects_x = []
    intersects_y = []

    n = len(straights)
    for i in range(0, n-1):
        for j in range(i+1, n):
            ri_m, ri_b = straights[i]
            rj_m, rj_b = straights[j]
            x = (rj_b - ri_b) / (ri_m - rj_m)
            y = (ri_b * rj_m - rj_b * ri_m) / (rj_m - ri_m)
            intersects_x.append(x)
            intersects_y.append(y)

    V = np.median(intersects_y)
    Km = np.median(intersects_x)
    # TODO: compute CIs
    res = res_object('Eisenthal-C.Bowden', V, Km, x=a, y=v0)
    # these are returned to help to draw a graph:
    res['intersections_x'] = intersects_x
    res['intersections_y'] = intersects_y
    res['straights_m'] = v0 / a
    res['straights_b'] = v0
    return res


def compute_methods(a, v0):
    # compute methods
    m_table = ({'name': 'L.-Burk',
                'method': lineweaver_burk,
                'color': 'g'},
               {'name': 'Hanes',
                'method': hanes_woolf,
                'color': 'c'},
               {'name': 'E.-Hofstee',
                'method': eadie_hofstee,
                'color': 'y'},
               {'name': 'Hyperbolic',
                'method': hyperbolic,
                'color': 'r'},
               {'name': 'C.-Bowden',
                'method': cornish_bowden,
                'color': 'k'})

    all_r = [m['method'](a, v0) for m in m_table]
    return MethodResults(a, v0, all_r)


# ------------ all methods object, with plots --------------------------
default_color_scheme = ('darkviolet',
                        'green',
                        'darkred',
                        'cornflowerblue',
                        'goldenrod')

class MethodResults(object):
    
    def __init__(self, a, v0, results_list):
        self.results=results_list
        self.a = a
        self.v0 = v0

    def __str__(self):
        results = self.results
        col_labels = ["Method                V      SE_V   Km     SE_Km"]
        fstring = '{:20} {:6.3f} {:6.3f} {:6.3f} {:6.3f}'
        mlines = []
        for r in results:
            if r['SE_V'] is None:
                SE_V = float('nan')
            else:
                SE_V = r['SE_V']
            if r['SE_Km'] is None:
                SE_Km = float('nan')
            else:
                SE_Km = r['SE_Km']
            mlines.append(fstring.format(r['name'], r['V'], SE_V, r['Km'], SE_Km))
        col_labels.extend(mlines)
        return '\n'.join(col_labels)
   
    def as_df(self):
        results = self.results
        index = []
        V = []
        SE_V = []
        Km = []
        SE_Km = []
        for r in results:
            index.append(r['name'])
            V.append(r['V'])
            SE_V.append(r['SE_V'])
            Km.append(r['Km'])
            SE_Km.append(r['SE_Km'])

        data = {'V': V, 'SE_V': SE_V, 'Km': Km, 'SE_Km': SE_Km}
        #print(data)
        df = pd.DataFrame(data, index=index)
        df.index.name = 'Method'
        return df


    def plot_hypers(self, colorscheme=None, 
                          title=None,
                          legend=True,
                          with_table=False,
                          grid=True):

        a = self.a
        v0 = self.v0
        all_results = self.results
        res_values = []
        f, ax = plt.subplots(1, 1, figsize=(8, 6))
        if colorscheme is None:
            colorscheme = default_color_scheme
        if title is None:
            title = 'All methods'

        xmax = max(a) * 1.1
        ymax = max(v0) * 1.1

        ax.set_title(title)
        ax.set_ylim(0, ymax)
        ax.set_xlim(0, xmax)

        for r, c in zip(all_results, colorscheme):
            V = r['V']
            Km = r['Km']
            x, y = MM_line(V, Km, xmax=xmax)

            ax.plot(x, y, label=r.name,
                          color=c,
                          linestyle='solid',
                          lw=2)
            res_values.append(res_tuple(r['name'], r['V'], r['Km'], r['SE_V'], r['SE_Km']))


        ax.plot(a, v0, marker='o',
                       linestyle='None', 
                       markerfacecolor='white', 
                       markeredgecolor='black', 
                       markeredgewidth=1.5, 
                       markersize=6)
        ax.set_xlabel('[S]')
        ax.set_ylabel('$v_0$', fontsize=18)
        if legend:
            ax.legend(loc='lower right')
        if grid:
            ax.grid()
        
        if with_table:
            # draw table
            col_labels = 'Method', 'V', 'SE_V', 'Km', 'SE_Km'
            the_table = ax.table(cellText=res_values,
                                 colWidths=[0.08, 0.05, 0.05, 0.05, 0.05],
                                 colLabels=col_labels,
                                 loc='upper left')
            the_table.set_fontsize(16)
            the_table.scale(2, 1.8)
            the_table.backgroundcolor='white'
            
        plt.show()
        return f

    def plot_others(self, colorscheme=None, grid=True):
        a = self.a
        v0 = self.v0
        all_r = self.results
        if colorscheme is None:
            colorscheme = default_color_scheme
        f, ax = plt.subplots(2, 2, figsize=(10,7.5))
        ax = [ax[0][0], ax[0][1], ax[1][0], ax[1][1]]
        for i in range(0,3):
             draw_lin_plot(ax[i], all_r[i], color=colorscheme[i])
        draw_cornish_bowden_plot(ax[3], all_r[4], color=colorscheme[4])
        f.tight_layout()

        plt.show()
        return f


def draw_lin_plot(ax, results, color='black', 
                               title=None,
                               grid=True):

    if title is None:
        title = results['name']
    x = results['x']
    y = results['y']

    xmax = max(x) * 1.1
    ymax = xmax * results['m'] + results['b']

    if results['m'] < 0:
        ytop = results['b']
    else:
        ytop = ymax
    ytop = 1.1 * ytop

    ax.set_title(title)
    ax.set_ylim(0, ytop)
    ax.set_xlim(0, xmax)
    
    ax.plot([0,xmax], [results['b'], ymax], color=color,
                  linestyle='solid',
                  lw=2)

    ax.plot(x, y,  marker='o',
                   linestyle='None', 
                   markerfacecolor='white', 
                   markeredgecolor='black', 
                   markeredgewidth=1.5, 
                   markersize=6)

    if results.name.startswith('Lineweaver'):
        xlabel = '1/[S]'
        ylabel = '$1/v_0$'
    elif results.name.startswith('Hanes'):
        xlabel = '[S]'
        ylabel = '[S]/$v_0$'
    elif results.name.startswith('Eadie'):
        xlabel = '$v_0$/[S]'
        ylabel = '$v_0$'
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

    a = results['x']
    v0 = results['y']
    intersections_x = results['intersections_x']
    intersections_y = results['intersections_y']
    if title is None:
        title = results['name']

    xmax = max(intersections_x) * 1.1
    ymax = max(intersections_y) * 1.1
    xmin = max(a) * 1.1
    ymin = 0.0

    ax.set_title(title)
    ax.set_ylim(0, ymax)
    ax.set_xlim(-xmin, xmax)

    for ai, v0i in zip(a, v0):
        ymaxi = v0i / ai * (xmax + ai)
        ax.plot([-ai, xmax], [0, ymaxi],
                  color='gray',
                  linestyle='solid',
                  lw=1)
    for x, y in zip(intersections_x, intersections_y):
        ax.plot(x, y,  
                   marker='o',
                   linestyle='None', 
                   markerfacecolor='white', 
                   markeredgecolor='black', 
                   markeredgewidth=1, 
                   markersize=4)
    ax.plot(results['Km'], results['V'], 
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

# constants
demo_data = """0.138 0.148
0.220 0.171
0.291 0.234
0.560 0.324
0.766 0.390
1.460 0.493
"""
empty_df = pd.DataFrame({'substrate': [0], 'rate':[0]})
empty_df.index.name = '#'

fig0 = Figure(figsize=(8, 6))
ax0 = fig0.subplots()
FigureCanvas(fig0)  # not needed for mpl >= 3.1
t = ax0.text(0.5, 0.5, 'no figure generated')

last_results = None
all_methods_list = ['Lineweaver-Burk', 'Hanes-Woolf', 'Eadie-Hofstee', 'Hyperbolic Reg.', 'Eisenthal-C.Bowden']

def read_data(data):  # used just for testing
    a = []
    v0 = []
    for line in data.splitlines():
        line = line.strip()
        if len(line) == 0:
            continue
        x1, x2 = line.split(None, 2)
        try:
            x1, x2 = float(x1), float(x2)
        except:
            continue
        a.append(x1)
        v0.append(x2)
    return np.array(a), np.array(v0)


def hypers_mpl(results, display_methods=all_methods_list,
                      colorscheme=None, 
                      title=None,
                      legend=True,
                      grid=True):

    default_color_scheme = ('darkviolet',
                            'green',
                            'darkred',
                            'cornflowerblue',
                            'goldenrod')
    a = results.a
    v0 = results.v0
    all_results = results.results
    res_values = []

    f = Figure(figsize=(8, 6))
    FigureCanvas(f) # not needed in mpl >= 3.1
    ax = f.add_subplot()
    
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
        if result['name'][:3] not in chosen3letter:
            continue

        x, y = MM_line(result['V'], result['Km'], xmax=xmax)

        ax.plot(x, y, label=result['name'],
                      color=color,
                      linestyle='solid',
                      lw=2)

    ax.plot(a, v0, marker='o',
                   linestyle='None', 
                   markerfacecolor='white', 
                   markeredgecolor='black', 
                   markeredgewidth=1.5, 
                   markersize=6)
    ax.set_xlabel('[S]')
    ax.set_ylabel('$v_0$', fontsize=18)
    if legend:
        ax.legend(loc='lower right')
    if grid:
        ax.grid()
    return f


def read_data_df(data_text):
    tf = StringIO(data_text)
    df = pd.read_csv(tf, delim_whitespace=True, comment='#', index_col=False)
    a = df.iloc[:, 0].values()
    v0 = df.iloc[:, 1].values()
    return a, v0


# widgetry

# data input
data_input_text = pn.widgets.input.TextAreaInput(height=400, width=300,
                                                 min_height=400,
                                                 min_width=100,
                                                 height_policy='min')

data_dfwidget = pn.widgets.DataFrame(empty_df, width=300, disabled=True)
#data_dfwidget = pn.pane.DataFrame(empty_df, width=400, index=False, justify='left')

data_input_column = pn.Column(data_input_text)

# data input buttons
clear_button = pn.widgets.Button(name='Clear', button_type='danger', width=80)
def b_clear(event):
    #data_input_column[0] = data_input_text
    data_input_text.value = data_input_text.placeholder
    edit_table_group.value = 'Edit'
    app_column[-1] = no_results_pane
clear_button.on_click(b_clear)

edit_table_group = pn.widgets.RadioButtonGroup(options=['Edit', 'Table'], width=100)
edit_table_group.value = 'Edit'

demo_button = pn.widgets.Button(name='Demo data', width=200)
def b_demo(event):
    data_input_text.value = demo_data
    edit_table_group.value = 'Edit'
demo_button.on_click(b_demo)

def transform_data2df(text):
    s, v0 = read_data(text)
    df = pd.DataFrame({'rate':v0}, index=s)
    df.index.name = 'substrate'
    return df
    
def change_data_view(event):
    if event.new == 'Table':
        df = transform_data2df(data_input_text.value)
        data_dfwidget.value = df
        data_input_column[0] = data_dfwidget
    else:
        data_input_column[0] = data_input_text

edit_table_group.param.watch(change_data_view, 'value')

# results
results_df = pn.widgets.Tabulator(empty_df)

mpl_pane = pn.pane.Matplotlib(fig0)

check_methods = pn.widgets.CheckBoxGroup(options=all_methods_list, value=all_methods_list, inline=False)

@pn.depends(check_methods)
def change_plot(check_methods):
    global last_results
    if last_results is not None:
        f = hypers_mpl(last_results, display_methods=check_methods)
        mpl_pane.object = f
    return mpl_pane

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

no_results_pane = pn.Column(pn.layout.Divider())


# the "Fit" button
fit_button = pn.widgets.Button(name='Fit', width=200, button_type='primary')
def b_fit(event):
    global last_results
    f = StringIO()
    #print(results_pane.pprint())
    s, v0 = read_data(data_input_text.value)
    print ('s  =', s, file=f)
    print ('v0 =', v0, file=f)
    print ('-------------------------------------------------', file=f)
    last_results = compute_methods(s, v0)
    df = last_results.as_df()
    print(df, file=f)
    newtxt = f.getvalue()

    results_df.value = df

    results_pane = pn.Column(pn.layout.Divider(), "## Fitting results", results_df, '## Plots',
                         pn.Row(change_plot, pn.Column(pn.Spacer(height=50), check_methods, fd_png, fd_pdf)))
    app_column[-1] = results_pane
fit_button.on_click(b_fit)

top_buttons = pn.Row(edit_table_group, clear_button)
data_input_row = pn.Row(data_input_column, pn.Column(top_buttons, demo_button, fit_button))

app_column = pn.Column("<br>\n# Michaelis-Menten equation fitting", "## Data input", data_input_row, no_results_pane)
app_column.servable()
# app_column
