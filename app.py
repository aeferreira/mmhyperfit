"""panel app to perform Michaelis-Menten equation fitting."""

from io import BytesIO

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import param

import panel as pn
from panel.widgets import Button

from mm_fitting.methods import compute_methods
from rwplots import (read_data,
                     report_str,
                     hypers_mpl,
                     plot_others_mpl,
                     PlotSettings)

APP_VERSION = "1.2"

pn.extension('tabulator', 'mathjax', 'floatpanel')


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

# widgetry ####################

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


# results

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

    hypers_mpl(results=None, ax=res_interface.f_ax['hypers_ax'])
    mpl_hypers.param.trigger('object')

    plot_others_mpl(results=None, f=res_interface.f_ax['others_f'])
    mpl_others.param.trigger('object')


clear_button = Button(name='Clear all', button_type='danger', width=150)
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
    n_points = len(subs_conc)
    if n_points <= 2:
        msg = '### <span style="color:red;">Error</span>\n\n'
        msg = msg + f'Not enough data points ({n_points}).\n\n'
        msg = msg + 'Must be at least 3.'
    elif np.var(subs_conc) / np.mean(subs_conc) < 1e-8:
        msg = '### <span style="color:red;">Error</span>\n\n'
        msg = msg + 'Substrate concentrations too close to perform fitting'
    else:
        msg = ''
    if msg:
        if app_display[-1] is not global_float:
            app_display.append(global_float)
        text = pn.pane.Markdown(msg)
        global_float.clear()
        global_float.append(text)
        global_float.theme = 'danger'
        # global_float.name = 'error'
        global_float.status = 'normalized'
        # pn.state.notifications.position = 'top-center'
        # pn.state.notifications.error(msg, duration=5000)
        return
    # make results_pane visible and compute results
    results_pane.visible = True
    res_interface.last_results = compute_methods(subs_conc, v0_values)

    # fill results text and trigger the drawing of new plots
    results_text.object = report_str(res_interface.last_results)
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
            dmsg = '### <span style="color:orange;">Warning:</span>\n\n'
            msg = dmsg + msg
            if app_display[-1] is not global_float:
                app_display.append(global_float)
            text = pn.pane.Markdown(msg)
            global_float.clear()
            global_float.append(text)
            global_float.theme = 'warning'
            global_float.satus = 'normalized'


edit_table_group.param.watch(change_data_view, 'value')

desc = pn.pane.Markdown(r"""
Fitting Michaelis-Menten equation to kinetic data using five methods

$$v_o = \\frac{V a}{K_m + a}$$

by António Ferreira, version """ + f'{APP_VERSION}', renderer='markdown')

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

data_input_row = pn.Row(pn.Spacer(width=10),
                        pn.WidgetBox(data_input_column,
                                     height=320,
                                     width=400),
                        pn.Column(fit_button, clear_button))


# plot settings widgets
parsetts = res_interface.plot_settings.param
method_choice = pn.widgets.MultiChoice.from_param(parsetts.include_methods)
display_legend = pn.widgets.Checkbox.from_param(parsetts.show_legend)
display_grid = pn.widgets.Checkbox.from_param(parsetts.show_grid)
display_Kms = pn.widgets.Checkbox.from_param(parsetts.show_Kms)
display_Vs = pn.widgets.Checkbox.from_param(parsetts.show_Vs)
line_color_choice = pn.widgets.ColorMap.from_param(parsetts.line_colors,
                                                   margin=(0, 0, 30, 10))


download_image = pn.widgets.FileDownload(callback=res_interface.hypers_file,
                                         label='Download image as',
                                         icon='download',
                                         button_style='solid',
                                         button_type='default',
                                         filename='hypers.pdf',)

image_format = pn.widgets.RadioButtonGroup(name='Image format',
                                           options=['png', 'svg'],
                                           button_style='outline',
                                           button_type='primary', height=40)


# TODO: use this idea sp that image download is triggered from a dropdown
# provocative_button = pn.widgets.Button(name='Other button', button_type='warning')
# def c(e):
#     download_image._clicks += 1

# provocative_button.on_click(c)
# ,provocative_button)
#   colors_button,
#   colors_hamburger)

# download_image.visible = False


plot_settings = pn.Column(pn.pane.Markdown('''#### Plot settings'''),
                          method_choice,
                          pn.widgets.StaticText(value='Show'),
                          pn.Row(display_legend,
                                 display_grid,
                                 display_Kms,
                                 display_Vs,),
                          line_color_choice,
                          pn.Row(download_image,
                                 image_format))

# plots
tabs = pn.Tabs(('MM equation plot', pn.Row(mpl_hypers, plot_settings)),
               ('Secondary plots', mpl_others))
plots_box = pn.WidgetBox(tabs)


float_config = {"headerControls": {"maximize": "remove", 'minimize': 'remove'}}
global_float = pn.layout.FloatPanel(pn.widgets.StaticText(value='Info'),
                                    name='', margin=20,
                                    config=float_config,
                                    contained=False, position='center')

results_pane = pn.Column(pn.layout.Divider(), "### Parameter values",
                         results_text,
                         pn.Spacer(height=50),
                         pn.Row(plots_box))

# start results pane hidden
results_pane.visible = False

# arrange components
app_title = 'Michaelis-Menten equation fitting'
# template = pn.template.VanillaTemplate(title=app_title)

# sidebar = pn.Column(desc,
#                     # fit_button,
#                     # clear_button
#                     )
# template.sidebar.append(sidebar)

custom_header_style = {
    'background_color': 'rgb(0,114,181)',
    'color': 'white',
    'padding': '10px',
    'margin': '0px 0px',
    # 'display': 'flex',
    'box-shadow': '5px 5px 20px silver'
}

header = pn.Row(pn.pane.Markdown('## Michaelis-Menten equation fitting',
                                 styles=custom_header_style,
                                 sizing_mode='stretch_width',
                                 width_policy='max'))

app_column = pn.Column(desc,
                       '### Data input',
                       data_input_row,
                       results_pane)

app_display = pn.Column(header, app_column)
# template.main.append(app_column)

# template.servable()
app_display.servable(title='Michaelis-Menten fitting')
# app_column
