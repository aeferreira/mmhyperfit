from __future__ import print_function
from methods import read_data, compute_methods

demo_data = """
    a     v
    0.138 0.148
    0.220 0.171
    0.291 0.234
    0.560 0.324
    0.766 0.390
    1.460 0.493
"""

s, v0 = read_data(demo_data)
print ('s  =', s)
print ('v0 =', v0)

results = compute_methods(s, v0)
print (results)

f = results.plot_hypers()
f.savefig('hypers.png')
f.savefig('hypers.pdf')

f = results.plot_others()
f.savefig('others.png')
f.savefig('others.pdf')
