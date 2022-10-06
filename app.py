# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from io import StringIO, BytesIO

from methods import compute_methods, MM_line, read_data
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas

import pandas as pd

import panel as pn
pn.extension()

# %% tags=[]
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
all_methods_list = ['LB', 'HW', 'EH', 'Hyp Reg', 'DLP']


# %%
# MM function drawing

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
    #if title is None:
        #title = 'All methods'

    xmax = max(a) * 1.1
    ymax = max(v0) * 1.1

    if title is not None:
        ax.set_title(title)
    ax.set_ylim(0, ymax)
    ax.set_xlim(0, xmax)
    skips = {'Han': 'HW', 'Lin': 'LB', 'Ead': 'EH', 'Hyp': 'Hyp Reg', 'Eis': 'DLP'}

    for r, c in zip(all_results, colorscheme):
        letters3 = r.name[:3]
        if skips[letters3] not in display_methods:
            continue
        V = r.V
        Km = r.Km
        x, y = MM_line(V, Km, xmax=xmax)

        ax.plot(x, y, label=r.name,
                      color=c,
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

def read_data_df(data_text):  # used just for testing
    tf = StringIO(data_text)
    df = read_csv(tf, delim_whitespace=True, comment='#', index_col=False)
    a = df.iloc[:, 0].values()
    v0 = df.iloc[:, 1].values()
    return a, v0


# %% tags=[]
# widgetry

# data input
data_input_text = pn.widgets.input.TextAreaInput(height=400, width=300,
                                                 min_height=400,
                                                 min_width=100,
                                                 height_policy='min')

data_dfwidget = pn.widgets.DataFrame(empty_df, width=300, disabled=True)
#data_dfwidget = pn.pane.DataFrame(empty_df, width=400, index=False, justify='left')

data_input_column = pn.Column(data_input_text)

# log
log_txt = pn.widgets.input.TextAreaInput(height=100, width=300,
                                                 min_height=100,
                                                 min_width=100,
                                                 height_policy='min')
log_txt.value = 'App started'

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
results_df = pn.widgets.DataFrame(empty_df, width=600, disabled=False)
mpl_pane = pn.pane.Matplotlib(fig0)

#check_methods = pn.widgets.CheckButtonGroup(name='Methods', value=all_methods_list, options=all_methods_list)
#check_methods = pn.widgets.MultiChoice(options=all_methods_list, value=all_methods_list,
                                       #width = 120,
                                       #margin=(0, 20, 0, 0))

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
    log_txt.value = log_txt.value + '\n' + newtxt
    results_df.value = df
    #mp = change_plot(check_methods)
    #results_pane[-1] = mp
    results_pane = pn.Column(pn.layout.Divider(), "## Fitting results", results_df, '## Plots',
                         pn.Row(change_plot, pn.Column(pn.Spacer(height=50), check_methods, fd_png, fd_pdf)))
    app_column[-1] = results_pane
fit_button.on_click(b_fit)

top_buttons = pn.Row(edit_table_group, clear_button)
data_input_row = pn.Row(data_input_column, pn.Column(top_buttons, demo_button, fit_button), log_txt)

#rw = pn.Row(app_column)
#f.savefig('hypers.png')
#f.savefig('hypers.pdf')

#f = results.plot_others()
#f.savefig('others.png')
#f.savefig('others.pdf')
app_column = pn.Column("# Michaelis-Menten equation fitting", "## Data input", data_input_row, no_results_pane)
app_column.servable()
#pn.serve(app_column)

# %%
