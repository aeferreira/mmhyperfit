"""Methods to compute Michaelis-Menten equation parameters and statistics."""

from io import BytesIO
from itertools import combinations
from collections import namedtuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

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
                         doc='standard error of the Michaelis constant')

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


def np_linregress(x, y):
    """Numpy linear regression almost compatible with scipy's linregress."""
    coefs, cov = np.polyfit(x, y, 1, cov=True)
    m, b = coefs
    Sm, Sb = np.diag(cov)**0.5
    r = np.corrcoef(x, y).ravel()[1]
    ret_dict = {'slope': m, 'intercept':
                b, 'rvalue': r, 'stderr':
                Sm, 'intercept_stderr': Sb}
    LinregressResult = namedtuple('LinregressResult', ret_dict.keys())
    return LinregressResult(**ret_dict)


RETURN_MSG = {1: 'Convergence in r.h.s. ("JtWdy")',
              2: 'Convergence in Parameters',
              -2: 'Maximum Number of Iterations Reached Without Convergence',
              -1: 'Residuals vector contains NaN or infinite values', }


def lm_matx(f, x, y, p_old, y_old, dX2, J, p, weight, dp, iteration):
    n_pars = len(p)
    pflat = p.ravel()
    y_hat = f(x, *pflat)
    my_func_calls = 1
    if dX2 > 0 or (iteration % (2 * n_pars) == 0):
        perturb = pflat.copy()
        J = np.zeros((len(y_hat), n_pars))
        delta = dp * (1 + np.abs(pflat))
        for j in range(n_pars):
            if delta[j] == 0.0:
                continue
            perturb[j] = pflat[j] + delta[j]
            y1 = f(x, *perturb)
            my_func_calls += 1
            perturb[j] = pflat[j] - delta[j]
            J[:, j] = (y1 - f(x, *perturb)) / (2 * delta[j])
            my_func_calls += 1
    else:
        h = p - p_old
        a = (np.array([y_hat - y_old]).T - J@h)@h.T
        b = h.T@h
        J = J + a/b
    delta_y = np.array([y - y_hat]).T
    chi_sq = (delta_y.T @ (delta_y * weight)).item()
    JtWJ = J.T @ (J * (weight * np.ones((1, n_pars))))
    JtWdy = J.T @ (weight * delta_y)
    return JtWJ, JtWdy, chi_sq, y_hat, J, my_func_calls


def lm(f, p, x, y,
       max_iter=1000,
       full_output=False):
    stopping_code = 0
    iteration = 0
    func_calls = 0
    n_pars = len(p)
    p = p.reshape(n_pars, 1)
    n_points = len(y)
    p_old = None
    y_old = None
    X2 = np.inf
    X2_old = X2

    J = None

    DoF = n_points - n_pars + 1

    if len(x) != len(y):
        raise ValueError('Mismatch of x and y lengths in data')
    if not all(np.isfinite(x)):
        raise ValueError('x vector contains NaN or infinite values')
    if not all(np.isfinite(y)):
        raise ValueError('y vector contains NaN or infinite values')

    weight = 1.0 / np.dot(y, y)
    dp = [1e-8]
    p_min, p_max = -100*abs(p), 100*abs(p)
    epsilon_1 = 1e-3   # convergence tolerance for gradient
    epsilon_2 = 1e-3   # convergence tolerance for parameters
    epsilon_4 = 1e-1   # determines acceptance of a L-M step
    lambda_0 = 1e-2    # initial value of damping parameter, lambda
    lambda_UP_fac = 11  # factor for increasing lambda
    lambda_DN_fac = 9   # factor for decreasing lambda
    if len(dp) == 1:
        dp = dp*np.ones(n_pars)

    stop = False  # termination flag
    if np.var(weight) == 0:
        weight = abs(weight) * np.ones((n_points, 1))
        # print('Using uniform weights for error analysis')
    else:
        weight = abs(weight)

    JtWJ, JtWdy, X2, y_hat, J, more_func_calls = lm_matx(f, x, y, p_old,
                                                         y_old, 1, J, p,
                                                         weight, dp, iteration)
    func_calls += more_func_calls

    lambda_0 = np.atleast_2d([lambda_0])
    iter_lambda = lambda_0
    X2_old = X2

    # initialize convergence history
    cvg_hst = np.ones((max_iter, n_pars+3))
    cvg_hst[0, 0:3] = func_calls, X2 / DoF, iter_lambda[0, 0]
    cvg_hst[0, 3:] = p[:, 0]

    while not stop and iteration <= max_iter:
        iteration += 1

        h = np.linalg.solve((JtWJ + iter_lambda*np.diag(np.diag(JtWJ))), JtWdy)

        p_try = p + h
        p_try = np.minimum(np.maximum(p_min, p_try), p_max)
        delta_y = (y - f(x, *p_try.ravel())).reshape(n_points, 1)
        if not all(np.isfinite(delta_y)):
            stopping_code = -1
            stop = True
            break
        func_calls += 1
        X2_try = delta_y.T @ (delta_y * weight)
        rho = (h.T @ (iter_lambda * h + JtWdy)) * 1.0 / (X2 - X2_try.item())

        if (rho > epsilon_4):
            dX2 = X2 - X2_old
            X2_old = X2
            p_old = p
            y_old = y_hat
            p = p_try

            (JtWJ, JtWdy, X2, y_hat,
             J, more_func_calls) = lm_matx(f, x, y, p_old, y_old,
                                           dX2, J, p,
                                           weight, dp, iteration)
            func_calls += more_func_calls
            iter_lambda = max(iter_lambda/lambda_DN_fac, 1.e-7)
        else:
            X2 = X2_old
            if iteration % (2*n_pars) == 0:
                (JtWJ, JtWdy, dX2,
                 y_hat, J, more_func_calls) = lm_matx(f, x, y, p_old,
                                                      y_old, -1, J, p,
                                                      weight, dp, iteration)
                func_calls += more_func_calls
            iter_lambda = min(iter_lambda*lambda_UP_fac, 1.e7)

        cvg_hst[iteration, 0:3] = func_calls, X2 / DoF, iter_lambda[0, 0]
        cvg_hst[iteration, 3:] = p[:, 0]

        if max(abs(JtWdy)) < epsilon_1 and iteration > 2:
            stopping_code = 1
            stop = True
        if max(abs(h)/(abs(p)+1e-12)) < epsilon_2 and iteration > 2:
            stopping_code = 2
            stop = True
        if iteration == max_iter:
            stopping_code = -2
            stop = True

    if np.var(weight) == 0:
        weight = DoF/(delta_y.T@delta_y) * np.ones((n_points, 1))

    redX2 = X2 / DoF

    (JtWJ, JtWdy, X2,
     y_hat, J, more_func_calls) = lm_matx(f, x, y, p_old, y_old,
                                          -1, J, p,
                                          weight, dp, iteration)
    func_calls += more_func_calls

    covar_p = np.linalg.inv(JtWJ)
    sigma_y = np.zeros((n_points, 1))
    for i in range(n_points):
        sigma_y[i, 0] = J[i, :] @ covar_p @ J[i, :].T
    sigma_y = np.sqrt(sigma_y)
    p = p.ravel()
    cvg_hst = cvg_hst[:iteration, :]
    if full_output:
        ret_dict = {'reduced_chisquare': redX2,
                    'sigma_y': sigma_y,
                    'iterations': cvg_hst.shape[0],
                    'history': cvg_hst}
        return p, covar_p, ret_dict, RETURN_MSG[stopping_code], stopping_code
    return p, covar_p


def lineweaver_burk(a, v0):
    """Compute parameters by Lineweaver-Burk linearization."""
    x, y = 1/a, 1/v0
    result = np_linregress(x, y)
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
    result = np_linregress(x, y)
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
    result = np_linregress(x, y)
    V = result.intercept
    Km = -result.slope
    SV = result.intercept_stderr
    SKm = result.stderr
    res = ResDict(method='Eadie-Hofstee',
                  V=V, Km=Km, SE_V=SV, SE_Km=SKm,
                  x=x, y=y, m=result.slope, b=result.intercept)
    return res


def hyperbolic(a, v0):
    """Compute parameters by non-linear least-squares regression."""
    try:
        # popt, pcov, *_ = curve_fit(MM, a, v0, p0=(max(v0), np.median(a)),
        #                            full_output=True,
        #                            check_finite=True)
        p_init = np.array([max(v0), np.median(a)])
        popt, pcov, *_ = lm(MM, p_init, a, v0, full_output=True)
        errors = np.sqrt(np.diag(pcov))
        V, Km = popt[0:2]
        SV, SKm = errors[0:2]
        res = ResDict(method='Hyperbolic regression',
                      V=V, Km=Km, SE_V=SV,
                      SE_Km=SKm, x=a, y=v0)
    except (ValueError, RuntimeError) as error:
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

plt.rc('mathtext', fontset='cm')


def hypers_mpl(results=None, ax=None,
               plot_settings=None,
               title=None,
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

    # plt.rc('mathtext', fontset='cm')
    # defaults
    colorscheme = default_color_scheme
    include_methods = all_methods_list
    show_legend = True
    # override with plot_settings
    if plot_settings is not None:
        include_methods = plot_settings.include_methods
        show_legend = plot_settings.show_legend

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
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if show_legend:
        ax.legend(loc='lower right')
    if grid:
        ax.grid(color="0.90")


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
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

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
        ax.grid(color="0.90")


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
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.axvline(x=0, color='black')
    ax.tick_params(axis='both', which='both', left=False, right=False)

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
    ax.set_xlabel('$Km$')
    ax.set_ylabel('$V$')
    if grid:
        ax.grid(color="0.90")


def read_data(data, return_msgs=False):
    a = []
    v0 = []
    use = []
    msgs = []
    allowed_nots = ('0', 'UNUSED', 'NO', 'FALSE', 'NOT', 'FALSE', 'NA', 'N/A')
    line_counter = 0
    for line in data.splitlines():
        line_counter += 1
        line = line.strip()
        if len(line) == 0:
            continue
        if line.startswith('#'):
            continue
        try:
            x1, x2, *x3 = line.split(None, 2)
        except ValueError:
            msgs.append(f'line {line_counter} skipped (too few values)')
            continue
        try:
            x1, x2 = float(x1), float(x2)
        except ValueError:
            if line_counter > 1:
                msgs.append(f'line {line_counter} skipped (not valid numbers)')
            continue
        datum_use = True
        if len(x3) > 0 and (x3[0].upper() in allowed_nots):
            datum_use = False
        a.append(x1)
        v0.append(x2)
        use.append(datum_use)
    if len(a) == 0:
        result = empty_df
    else:
        result = pd.DataFrame({'substrate': a,
                               'rate': v0,
                               'use': use})
    msg = msg = '\n'.join(msgs) if len(msgs) > 0 else ''
    if return_msgs:
        return result, msg
    else:
        return result


def data_as_txt(df):
    lines = ['substrate      rate']
    for lbl, row in df.iterrows():
        used = '  unused' if not row.use else ''
        line = f'{row.substrate:.5f}   {row.rate:.5f}{used}'
        lines.append(line)
    return '\n'.join(lines)


# widgetry


empty_df = pd.DataFrame({'substrate': [0.1],
                         'rate': [0.1],
                         'use': [True]})
empty_df.index.name = '#'

# data input widget (a Tabulator widget)

formatters = {'use': {'type': 'tickCross'}, }

editors = {'use': 'tickCross',
           'substrate': {'type': 'number', 'max': 1000, 'step': 0.01,
                         'verticalNavigation': 'table'},
           'rate': {'type': 'number', 'max': 1000, 'step': 0.01,
                    'verticalNavigation': 'table'}}

data_input = pn.widgets.Tabulator(empty_df, width=400, show_index=False,
                                  selectable='checkbox',
                                  widths={'substrate': 130, 'rate': 130},
                                  text_align={'substrate': 'left',
                                              'rate': 'left',
                                              'use': 'center'},
                                  formatters=formatters,
                                  editors=editors)

data_input_text = pn.widgets.input.TextAreaInput(width=350,
                                                 sizing_mode='stretch_height',
                                                 min_height=220)

# data input buttons

add_row_button = Button(name='+', width=20, align=('end', 'center'))


def add_row(event):
    new_row = pd.DataFrame({'substrate': [0.1], 'rate': [0.1], 'use': [True]})
    newdf = pd.concat([data_input.value, new_row], ignore_index=True)
    data_input.value = newdf


add_row_button.on_click(add_row)

# Edit menu

menu_items = [('Invert selection', 'invert_selection'),
              ('Select not "used"', 'select_not_used'),
              ('Toggle "use"', 'toggle_use'),
              None,
              ('Delete selected', 'del_selected'),
              ('Clear all data', 'del_all')]


# handling the edit menu

def handle_edit(event):
    choice = event.new
    df = data_input.value
    selection = data_input.selection
    not_selected = pd.Series([True]*len(df))
    not_selected[selection] = False
    if choice == 'del_all':
        data_input.value = empty_df
    elif choice == 'toggle_use':
        if len(selection) == 0:
            not_selected = ~not_selected
        new_use = df.use.where(not_selected, ~df.use)
        data_input.value = df.assign(use=new_use)
    elif choice == 'invert_selection':
        inverted = [i for i in range(len(df)) if i not in selection]
        data_input.selection = inverted
    elif choice == 'select_not_used':
        not_used = [i for (i, used) in enumerate(df.use) if not used]
        data_input.selection = not_used
    elif choice == 'del_selected':
        data_input.value = df[not_selected]
    elif choice == 'paste':
        pass
    else:
        pass  # do nothing


select_button = pn.widgets.MenuButton(name='Select', width=120,
                                      items=menu_items,
                                      button_type='success')
select_button.on_click(handle_edit)

# txt vs DataFrame view

edit_table_group = pn.widgets.RadioButtonGroup(options=['table', 'text'],
                                               width=100)
edit_table_group.value = 'table'

# demo data

# this is data from
# [Atkinson, M.R., Jackson, J.F., Morton, R.K.(1961) Biochem. J. 80(2):318-23]
# (https://doi.org/10.1042/bj0800318),
# used for method comparison by
# [Wilkinson, G.N. (1961) Biochem. J. 80(2):324–332]
# (https://doi.org/10.1042/bj0800324)

DEMO_DATA = """# https://doi.org/10.1042/bj0800318
0.138 0.148
0.220 0.171
0.291 0.234
0.560 0.324
0.766 0.390
1.460 0.493
"""


def b_demo(event):
    data_input.value = read_data(DEMO_DATA)


demo_button = Button(name='Demo')
demo_button.on_click(b_demo)


# def change_data_view(event):
#     if event.new == 'Check':
#         a, v0 = read_data(data_input_text.value)
#         df = pd.DataFrame({'rate': v0}, index=a)
#         df.index.name = 'substrate'

#         data_input.value = df
#         data_input_column[0] = data_input
#         pn.state.notifications.position = 'top-left'
#         pn.state.notifications.info('Data OK', duration=3000)
#     else:
#         data_input_column[0] = data_input_text


# results

class PlotSettings(param.Parameterized):
    include_methods = param.ListSelector(default=list(all_methods_list),
                                         objects=list(all_methods_list))
    show_legend = param.Boolean(default=True, label='Legend')


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


# reset button
def b_reset(event):
    results_pane.visible = False
    data_input.value = empty_df
    data_input_text.value = ''
    edit_table_group.value = 'table'
    results_text.object = ''
    # data_input_text.value = data_input_text.placeholder

    hypers_mpl(results=None, ax=res_interface.f_ax['hypers_ax'])
    mpl_hypers.param.trigger('object')

    plot_others_mpl(results=None, f=res_interface.f_ax['others_f'])
    mpl_others.param.trigger('object')


clear_button = Button(name='Reset', button_type='danger', width=150)
clear_button.on_click(b_reset)


# "Fit" button
def b_fit(event):
    # make results_pane visible and draw
    results_pane.visible = True

    # compute results
    df = data_input.value
    df = df[df.use]
    # TODO: validate
    subs_conc = df['substrate'].values
    v0_values = df['rate'].values
    res_interface.last_results = compute_methods(subs_conc, v0_values)

    # fill results text and trigger the drawing of new plots
    results_text.object = report_str(res_interface.last_results['results'])
    res_interface.e = True


fit_button = Button(name='Fit', width=150,
                    button_type='primary',
                    icon='calculator', icon_size='2em')
fit_button.on_click(b_fit)

edit_buttons = pn.Row(edit_table_group,
                      #   paste_button,
                      demo_button,
                      add_row_button,
                      select_button,)

data_input_column = pn.Column(edit_buttons, data_input, height=250)
# data_input_column = pn.Column(data_input_text, height=200)


def change_data_view(event):
    if event.new == 'text':
        data_input_text.value = data_as_txt(data_input.value)
        demo_button.disabled = True
        add_row_button.disabled = True
        select_button.disabled = True
        data_input_column[1] = data_input_text
    else:
        df, msg = read_data(data_input_text.value, return_msgs=True)
        data_input.value = df
        demo_button.disabled = False
        add_row_button.disabled = False
        select_button.disabled = False
        data_input_column[1] = data_input
        if msg:
            pn.state.notifications.position = 'top-center'
            pn.state.notifications.warning(msg, duration=5000)


edit_table_group.param.watch(change_data_view, 'value')

header = pn.pane.Markdown(r"""
#### Fitting Michaelis-Menten equation to kinetic data using seven methods

$$v_o = \\frac{V a}{K_m + a}$$

by António Ferreira

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

data_input_row = pn.Row(pn.WidgetBox(data_input_column,
                                     height=320,
                                     width=400),)

# plot settings
method_choice = CheckBoxGroup.from_param(res_interface.
                                         plot_settings.param.
                                         include_methods,
                                         inline=False)

display_legend = pn.widgets.Checkbox.from_param(res_interface.
                                         plot_settings.param.
                                         show_legend,
                                         inline=False)

download_png = FileDownload(callback=res_interface.get_png_hypers,
                            filename='hypers.png', width=200)
download_pdf = FileDownload(callback=res_interface.get_pdf_hypers,
                            filename='hypers.pdf', width=200)

plot_settings = pn.Column(pn.pane.Markdown('''#### Plot settings'''),
                          method_choice,
                          display_legend,
                          download_png,
                          download_pdf)

# plots
tabs = pn.Tabs(('MM equation plot', pn.Row(mpl_hypers, plot_settings)),
               ('Secondary plots', mpl_others))
plots_box = pn.WidgetBox(tabs)

results_pane = pn.Column(pn.layout.Divider(), "### Parameter values",
                         results_text,
                         pn.Spacer(height=50),
                         pn.Row(plots_box))

# start results pane hidden
results_pane.visible = False

# arrange components in template
app_title = 'Michaelis-Menten equation fitting'
template = pn.template.VanillaTemplate(title=app_title)


sidebar = pn.Column(header, fit_button, clear_button)
template.sidebar.append(sidebar)

app_column = pn.Column('### Data input',
                       data_input_row, results_pane)
template.main.append(app_column)

template.servable()
# app_column.servable(title='Michaelis-Menten fitting')
# app_column
