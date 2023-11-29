import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mm_fitting.methods import MM


# ------------- (str) report of results ------

def repr_x_deltax(value, delta):
    if delta is None:
        return f'{value:6.3f}'
    return f"{value:6.3f} ± {delta:6.3f}"


def report_str(results):
    results = results['results']
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
               title=None):

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

    chosen3letter = [choice[:3] for choice in plot_settings.include_methods]

    colors = {}
    kept = []
    for result, color in zip(all_results, colorscheme):
        if result.method[:3] in chosen3letter:
            colors[result.method] = color
            kept.append(result)

    maxKm = max([result.Km for result in kept])
    maxV = max([result.V for result in kept])
    maxVover2 = max([result.V/2.0 for result in kept])

    if plot_settings.show_Vs:
        ymax = max(max(v0), maxV)
    else:
        if plot_settings.show_Kms:
            ymax = max(max(v0), maxVover2)
        else:
            ymax = max(v0)
    ymax = ymax * 1.1

    if plot_settings.show_Kms:
        xmax = max(max(a), maxKm)
    else:
        xmax = max(a)
    xmax = xmax * 1.1

    ax.set_ylim(0, ymax)
    ax.set_xlim(0, xmax)

    if title is not None:
        ax.set_title(title)

    for result in kept:

        V, Km = result.V, result.Km
        line_x = np.linspace(0.0, xmax, 200)
        line_y = MM(line_x, V, Km)

        color = colors[result.method]

        ax.plot(line_x, line_y, label=result.method,
                color=color, linestyle='solid', lw=2, zorder=4.0)
        if plot_settings.show_Kms:
            line_x = (0.0, Km, Km)
            line_y = (V / 2.0, V / 2.0, 0.0)
            ax.plot(line_x, line_y, color=color,
                    linestyle='solid', lw=0.8,
                    marker='o', markersize=3, clip_on=False, zorder=3.5)
        if plot_settings.show_Vs:
            ax.axhline(y=V, color=color, linestyle='solid', lw=0.8, zorder=3.5)

    ax.plot(a, v0, marker='o',
            linestyle='None',
            markerfacecolor='white',
            markeredgecolor='black',
            markeredgewidth=1.5,
            markersize=6, zorder=5)
    ax.set_xlabel('$a$', fontsize=16)
    ax.set_ylabel('$v_o$', fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if plot_settings.show_legend:
        ax.legend(loc='lower right', fontsize="8")
    if plot_settings.show_grid:
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


empty_df = pd.DataFrame({'substrate': pd.Series([], dtype='float'),
                         'rate': pd.Series([], dtype='float'),
                         'use': pd.Series([], dtype='bool')})
empty_df.index.name = '#'
