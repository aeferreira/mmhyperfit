"""panel app to perform Michaelis-Menten equation fitting."""

from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import param

import panel as pn
from panel.widgets import Button

from mm_fitting.methods import MM, compute_methods

APP_VERSION = "1.0"

pn.extension('tabulator', 'mathjax')
pn.extension(notifications=True)


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

    maxKm = max([result.Km for result in all_results])
    maxV = max([result.V for result in all_results])
    xmax = max(max(a), maxKm) * 1.1
    if plot_settings.show_Vs:
        ymax = max(max(v0), maxV)
    else:
        ymax = max(v0)
    ymax = ymax * 1.1

    if title is not None:
        ax.set_title(title)
    ax.set_ylim(0, ymax)
    ax.set_xlim(0, xmax)
    chosen3letter = [choice[:3] for choice in plot_settings.include_methods]

    for result, color in zip(all_results, colorscheme):
        if result.method[:3] not in chosen3letter:
            continue

        V, Km = result.V, result.Km
        line_x = np.linspace(0.0, xmax, 200)
        line_y = MM(line_x, V, Km)

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


# widgetry


empty_df = pd.DataFrame({'substrate': pd.Series([], dtype='float'),
                         'rate': pd.Series([], dtype='float'),
                         'use': pd.Series([], dtype='bool')})
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
    show_grid = param.Boolean(default=False, label='Grid')
    show_Kms = param.Boolean(default=False, label='Km')
    show_Vs = param.Boolean(default=False, label='V')


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

    def hypers_file(self):
        if self.last_results is not None:
            image_type = image_format.value
            hypers_mpl(self.last_results, ax=self.f_ax['hypers_ax'],
                       plot_settings=self.plot_settings)
            f = self.f_ax['hypers_f']
            bio = BytesIO()
            download_image.filename = f'hypers.{image_type}'
            if image_type == 'png':
                f.savefig(bio, format='png', dpi=150)
            elif image_type == 'pdf':
                f.savefig(bio, format='pdf')
            elif image_type == 'svg':
                f.savefig(bio, format='svg', dpi=150)
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


clear_button = Button(name='Clear', button_type='danger', width=150)
clear_button.on_click(b_reset)


# "Fit" button
def b_fit(event):

    # compute results
    edit_table_group.value = 'table'
    df = data_input.value
    df = df[df.use]
    subs_conc = df['substrate'].values
    v0_values = df['rate'].values
    # validate data
    if len(subs_conc) <= 2:
        msg = 'Too few points (must be at least 3)'
    elif np.var(subs_conc) / np.mean(subs_conc) < 1e-8:
        msg = 'substrate concentrations too close to perform fitting'
    else:
        msg = ''
    if msg:
        pn.state.notifications.position = 'top-center'
        pn.state.notifications.error(msg, duration=5000)
        return
    # make results_pane visible and compute results
    results_pane.visible = True
    res_interface.last_results = compute_methods(subs_conc, v0_values)

    # fill results text and trigger the drawing of new plots
    results_text.object = report_str(res_interface.last_results['results'])
    res_interface.e = True


fit_button = Button(name='Fit', width=150,
                    button_type='primary',
                    icon='calculator', icon_size='2em')
fit_button.on_click(b_fit)

edit_buttons = pn.Row(edit_table_group,
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

desc = pn.pane.Markdown(r"""
#### Fitting Michaelis-Menten equation to kinetic data using five methods

$$v_o = \\frac{V a}{K_m + a}$$

by António Ferreira

Version """ + f'{APP_VERSION}', renderer='markdown')

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
parsetts = res_interface.plot_settings.param
method_choice = pn.widgets.MultiChoice.from_param(parsetts.include_methods)
display_legend = pn.widgets.Checkbox.from_param(parsetts.show_legend)
display_grid = pn.widgets.Checkbox.from_param(parsetts.show_grid)
display_Kms = pn.widgets.Checkbox.from_param(parsetts.show_Kms)
display_Vs = pn.widgets.Checkbox.from_param(parsetts.show_Vs)

download_image = pn.widgets.FileDownload(callback=res_interface.hypers_file,
                                         label='Download image as',
                                         icon='download',
                                         filename='hypers.pdf',
                                         height=30)

image_format = pn.widgets.RadioButtonGroup(name='Image format',
                                           options=['png', 'svg'],
                                           button_style='outline',
                                           button_type='primary')

plot_settings = pn.Column(pn.pane.Markdown('''#### Plot settings'''),
                          method_choice,
                          pn.widgets.StaticText(value='Show'),
                          pn.Row(display_legend,
                                 display_grid,
                                 display_Kms,
                                 display_Vs),
                          pn.Row(download_image, image_format), )

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

sidebar = pn.Column(desc,
                    fit_button,
                    clear_button)
template.sidebar.append(sidebar)

app_column = pn.Column('### Data input',
                       data_input_row, results_pane)
template.main.append(app_column)

template.servable()
# app_column.servable(title='Michaelis-Menten fitting')
# app_column
