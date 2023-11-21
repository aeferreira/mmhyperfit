"""Methods to compute Michaelis-Menten equation parameters and statistics."""

from itertools import combinations
from collections import namedtuple

import numpy as np

# namedtuple class to contain results

field_names = 'method error V Km SE_V SE_Km m b x y intersections dlp_lines'
field_defaults = ('Hanes', None, 0.0, 0.0, None, None,
                  None, None, None, None, None, None)
ResTuple = namedtuple('ResTuple', field_names,
                      rename=True,
                      defaults=field_defaults)

# ------------ fitting methods --------------------------
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
    p_old, y_old = None, None
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
    epsilon_1 = 1e-3    # convergence tolerance for gradient
    epsilon_2 = 1e-3    # convergence tolerance for parameters
    epsilon_4 = 1e-1    # determines acceptance of a L-M step
    iter_lambda = 1e-2  # initial value of damping parameter, lambda
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

    # lambda_0 = np.atleast_2d([lambda_0])
    # iter_lambda = lambda_0

    X2_old = X2

    # initialize convergence history
    cvg_hst = np.ones((max_iter, n_pars+3))
    cvg_hst[0, 0:3] = func_calls, X2 / DoF, iter_lambda  # [0, 0]
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

        cvg_hst[iteration, 0:3] = func_calls, X2 / DoF, iter_lambda  # [0, 0]
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
    res = ResTuple(method='Lineweaver-Burk',
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
    res = ResTuple(method='Hanes',
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
    res = ResTuple(method='Eadie-Hofstee',
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
        res = ResTuple(method='Hyperbolic regression',
                       V=V, Km=Km, SE_V=SV,
                       SE_Km=SKm, x=a, y=v0)
    except (ValueError, RuntimeError) as error:
        res = ResTuple(method='Hyperbolic regression',
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
        res = ResTuple(method='Eisenthal-C.Bowden',
                       V=V, Km=Km, x=a, y=v0,
                       intersections=intersects,
                       dlp_lines=np.array(straights))
    except ValueError as error:
        res = ResTuple(method='Eisenthal-C.Bowden', x=a, y=v0,
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


def show_results(results):
    a = results['a']
    v0 = results['v0']
    res = results['results']
    print('-- Data points ---')
    for subst, rate in zip(a, v0):
        print(f'{subst:.6f} {rate:.6f}')
    print('-- Results ---')
    if res is None:
        print('NO RESULTS WERE GENERATED')
        return
    for method in res:
        print(method.method)
        if method.error is not None:
            print('ERROR')
            print(method.error)
            continue
        print('Vmax =', method.V)
        print('Km   =', method.Km)


def read_from_whitespace(text):
    a = []
    v = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) == 0:
            continue
        subs, rate = line.split(None, 1)
        a.append(float(subs))
        v.append(float(rate))
    return np.array(a), np.array(v)


if __name__ == '__main__':
    a = np.array([0.138, 0.22, 0.291, 0.56, 0.766, 1.46])
    v = np.array([0.148, 0.171, 0.234, 0.324, 0.39, 0.493])
    res = compute_methods(a, v)
    show_results(res)
    a = np.array([0.138, 0.22, 0.291, 0.56, 0.766, 1.46])
    v = np.array([0.05, 0.1, 0.234, 0.324, 0.39, 0.493])
    res = compute_methods(a, v)
    show_results(res)
    problem3 = '''
    0.212789435	0.000205
    0.411679048	0.000351
    0.596798774	0.000412
    0.768453499	0.000758
    1.073340039	0.000783
    '''
    a, v = read_from_whitespace(problem3)
    res = compute_methods(a, v)
    show_results(res)

    problem4 = '''
    0.239         0.01074
    0.412         0.01932
    0.597         0.02616
    0.768         0.03552
    1.073         0.04410'''
    a, v = read_from_whitespace(problem4)
    res = compute_methods(a, v)
    show_results(res)

    problem5 = '''
    0.80000   0.05000
    2.00000   0.10000
    4.00000   0.43300
    6.00000   0.48800
    20.00000   0.62000'''
    a, v = read_from_whitespace(problem5)
    res = compute_methods(a, v)
    show_results(res)
