from __future__ import print_function
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress as lreg
import matplotlib.pyplot as pl


"""Methods to determine Michaelis-Menten equation parameters and statistics.

   Wilkinson [TODO: insert reference] data is used to test the methods."""


class Kin_results(object):
    """Object that holds data from a computation, supporting dot access.

       Mandatory members are:

       name - Name of the method used (str).
       error - None by default, str describing error in computation
       V - limiting rate (float)
       Km - Michaelis constant (float)
       SS - sum of squares of residuals to the Michelis-Menten equation (float)

       v_hat - estimated rate values (iterable of floats)

       Optional, depending on the method:
       SE_V - standard error of the limiting rate
       SE_Km - standard error of the Michelis constant

       Optional for linearizations:
       x - x-values during linearization (iterable of floats)
       y - y-values during linearization (iterable of floats)
       m - slope of linearization
       b - intercept of linearization"""

    def __init__(self, name):
        self.name = name
        self.error = None
        self.V = 0.0
        self.Km = 0.0
        self.SS = 0.0
        self.v_hat = None

# ------------ util functions --------------------------


def lists2arrays(x, y):
    return (np.array(x), np.array(y))


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
    x = np.linspace(x0, xmax, 100)
    return x, MM(x, V, Km)


def res_tuple(method, V, Km, SV=None, SKm=None):
    sV = "{:6.4f}".format(V)
    sKm = "{:6.4f}".format(Km)
    if SV is None:
        sSV = 'n/a'
    else:
        sSV = "{:6.4f}".format(SV)
    if SKm is None:
        sSKm = 'n/a'
    else:
        sSKm = "{:6.4f}".format(SKm)
    return method, sV, sSV, sKm, sSKm


def res_object(method, V, Km, SE_V=None, SE_Km=None, error=None,
               x=None, y=None, m=None, b=None):
    r = Kin_results(method)
    r.V = V
    r.Km = Km
    r.SE_V = SE_V
    r.SE_Km = SE_Km
    r.error = error
    r.x = x
    r.y = y
    r.m = m
    r.b = b
    return r


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
    res.intersections_x = intersects_x
    res.intersections_y = intersects_y
    res.straights_m = v0 / a
    res.straights_b = v0
    return res


def compute_methods(a, v0):
    # compute and plot lines
    all_r = []

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

    for m in m_table:
        r = m['method'](a, v0)
        all_r.append(r)
    ret = MethodResults(a, v0, all_r)
    return ret


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
            if r.SE_V is None:
                SE_V = float('nan')
            else:
                SE_V = r.SE_V
            if r.SE_Km is None:
                SE_Km = float('nan')
            else:
                SE_Km = r.SE_Km
            mlines.append(fstring.format(r.name, r.V, SE_V, r.Km, SE_Km))
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
            index.append(r.name)
            V.append(r.V)
            SE_V.append(r.SE_V)
            Km.append(r.Km)
            SE_Km.append(r.SE_Km)

        data = {'V': V, 'SE_V': SE_V, 'Km': Km, 'SE_Km': SE_Km}
        print(data)
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
        f, ax = pl.subplots(1, 1, figsize=(8, 6))
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
            V = r.V
            Km = r.Km
            x, y = MM_line(V, Km, xmax=xmax)

            ax.plot(x, y, label=r.name,
                          color=c,
                          linestyle='solid',
                          lw=2)
            res_values.append(res_tuple(r.name, r.V, r.Km, r.SE_V, r.SE_Km))


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
            
        pl.show()
        return f

    def plot_others(self, colorscheme=None, grid=True):
        a = self.a
        v0 = self.v0
        all_r = self.results
        if colorscheme is None:
            colorscheme = default_color_scheme
        f, ax = pl.subplots(2, 2, figsize=(10,7.5))
        ax = [ax[0][0], ax[0][1], ax[1][0], ax[1][1]]
        for i in range(0,3):
             draw_lin_plot(ax[i], all_r[i], color=colorscheme[i])
        draw_cornish_bowden_plot(ax[3], all_r[4], color=colorscheme[4])
        f.tight_layout()

        pl.show()
        return f


def draw_lin_plot(ax, results, color='black', 
                               title=None,
                               grid=True):

    if title is None:
        title = results.name
    x = results.x
    y = results.y

    xmax = max(x) * 1.1
    ymax = xmax * results.m + results.b

    if results.m < 0:
        ytop = results.b
    else:
        ytop = ymax
    ytop = 1.1 * ytop

    ax.set_title(title)
    ax.set_ylim(0, ytop)
    ax.set_xlim(0, xmax)
    
    ax.plot([0,xmax], [results.b, ymax], color=color,
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

    a = results.x
    v0 = results.y
    intersections_x = results.intersections_x
    intersections_y = results.intersections_y
    if title is None:
        title = results.name

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


def read_data(wilkinson):  # used just for testing
    wdata = [w.strip() for w in wilkinson.splitlines()]
    a = []
    v0 = []
    for i in wdata:
        if len(i) == 0:
            continue
        x1, x2 = i.split(None, 2)
        try:
            x1 = float(x1)
            x2 = float(x2)
        except:
            continue
        a.append(x1)
        v0.append(x2)
    a = np.array(a)
    v0 = np.array(v0)
    return a, v0


if __name__ == '__main__':
    wilkinson = """
    a     v
    0.138 0.148
    0.220 0.171
    0.291 0.234
    0.560 0.324
    0.766 0.390
    1.460 0.493
    """
    a, v0 = read_data(wilkinson)

    print ('a  =', a)
    print ('v0 =', v0)

    results = compute_methods(a, v0)    
    print (results)
    print(results.as_df())
    
    results.plot_hypers(with_table=True)
    
    results.plot_others()

    pl.show()
    