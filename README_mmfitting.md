# Michaelis-Menten equation fitting to data

A Python module that implements five methods to fit the Michaelis-menten equation to a set of points of rate vs substrate concentration.

The methods are:

- Non-linear regression (Levemberg-Marquard algorithm applied to Michaelis-Menten equation)
- The Direct Linear Plot
- The Lineweaver-Burk linearization
- The Hanes linearization
- The Eddie-Hofstee linearization

### Short usage:

Given two `numpy` 1D-arrays, `a` and `v0` containing substrate concentrations and initial rates, respectively,

``` python3
results = mm_fitting.compute_methods(a, v0)
```

will apply all five methods and generate a `dict` with keys `a` and `v0` and `results`. The value of
`results` will be a list of namedtuples containg the results for each method.

### `numpy`-only dependency

All methods are implemented in numpy and do not require either `scipy` or data analysis module like `pandas`:

- linearizations are computed by a thin wrapper of `numpy.polyfit()` with degree one.
- non-linear regression is computed using a numpy-only version of the Levemberg-Marquard algorithm. Code was adapted from Abner Bogan's Github repo for a numpy-only version of the algorithm ([abnerbog/
levenberg-marquardt-method
](https://github.com/abnerbog/levenberg-marquardt-method)).