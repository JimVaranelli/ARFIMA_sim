import sys
import numpy as np
from scipy.ndimage.interpolation import shift
from numpy.testing import assert_equal, assert_almost_equal
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.stattools import adfuller

# binomial expansion for ARFIMA models
def _calc_arfima_binomial(n, nterms):
    # switch equation side
    n = -n
    bc = np.zeros([nterms, 1])
    bc[0] = 1
    # generate coefficients
    for i in range(1, nterms):
        bc[i] = abs(bc[i - 1] * (n - (i - 1)) / i)
    return bc


def ARFIMA_sim(p_coeffs, q_coeffs, d, slen, alpha=0, sigma=1, numseas=100):
    """
    Generate a random ARFIMA(p,d,q) series. Generalizes to ARMA(p,q)
    when d = 0, and ARIMA(p,d,q) when d = 1.
        
    User provides an array of coefficients for the AR(p) and MA(q)
    portions of the series as well as the fractional differencing
    parameter and the required length. A constant may optionally be
    specified, as well as the standard deviation of the Gaussian
    innovations, and the number of seasoning samples to be
    generated before recording the series.

    Parameters
    ----------
    p_coeffs : array_like
        AR(p) coefficients
        len(p_coeffs) <= 10
    q_coeffs : array_like
        MA(q) coefficients
        len(q_coeffs) <= 10
    d : float
        fractional differencing parameter
        -1 < d <= 1
    slen : int
        number of samples in output ARFIMA series
        10 <= len(series) <= 100000
    alpha : float
        series constant (default=0)
    sigma : float
        standard deviation of innovations
    numseas : int
        number of seasoning samples (default=100)
        0 <= num(seasoning) <= 10000

    Returns
    -------
    series : 1d array
        random ARFIMA(p,d,q) series of specified length

    Notes
    -----
    MA(q) parameters follow the Box-Jenkins convention which uses a
    difference representation for the MA(q) process which is the opposite
    of the standard ARIMA MA(q) summation representation. This matches the
    operation of SAS/farmasim and R/arfimasim. As such, the SAS/farmafit
    and R/arfima MA(q) estimates match the sign of the specified MA(q)
    parameters while the statsmodels ARIMA().fit() estimates have opposite
    the specified MA(q) parameter signs.

    References
    ----------
    SAS Institute Inc (2013). SAS/IML User's Guide. Cary, NC: SAS Institute
    Inc.

    Veenstra, J.Q. (2012). Persistence and Anti-persistence: Theory and
    Software (Doctoral Dissertation). Western University, Ontario, Canada.
    """
    p = np.asarray(p_coeffs)
    if p.ndim > 2 or (p.ndim == 2 and p.shape[1] != 1):
        raise ValueError(
            'ARFIMA_sim: p must be a 1d array or a 2d array with a single column')
    p = np.reshape(p, (-1, 1))
    if p.shape[0] > 10:
        raise ValueError(
            'ARFIMA_sim: AR order must be <= 10')
    q = np.asarray(q_coeffs)
    if q.ndim > 2 or (q.ndim == 2 and q.shape[1] != 1):
        raise ValueError(
            'ARFIMA_sim: q must be a 1d array or a 2d array with a single column')
    q = np.reshape(q, (-1, 1))
    if q.shape[0] > 10:
        raise ValueError(
            'ARFIMA_sim: MA order must be <= 10')
    if d <= -1 or d > 1:
        raise ValueError(
            'ARFIMA_sim: valid differencing parameter must be in range (-1, 1]')
    if slen < 10 or slen > 100000:
        raise ValueError(
            'ARFIMA_sim: valid series length must be in range [10, 100000]')
    if numseas < 0 or numseas > 10000:
        raise ValueError(
            'ARFIMA_sim: valid seasoning length must be in range [0, 10000]')
    # check for negative fractional d. if negative,
    # add a unity order of integration, then single
    # difference the final series.
    neg = 0
    if d < 0:
        d += 1
        neg = 1
    # generate the MA(q) series
    lqc = q.shape[0]
    if lqc == 0:
        ma = np.random.normal(scale=sigma, size=slen+numseas)
    else:
        e = np.random.normal(scale=sigma, size=slen+numseas)
        ma = np.zeros([slen+numseas, 1])
        ma[0] = e[0]
        for t in range (1, slen + numseas):
            err = e[max(0,t-lqc):t]
            qcr = np.flip(q[0:min(lqc, t)])
            ma[t] = e[t] - np.dot(err, qcr)
    # generate the ARMA(p,q) series
    lpc = p.shape[0]
    if lpc == 0:
        arma = ma
    else:
        arma = np.zeros([slen+numseas, 1])
        arma[0] = ma[0]
        for t in range (1, slen + numseas):
            arr = arma[max(0,t-lpc):t]
            pcr = np.flip(p[0:min(lpc, t)])
            arma[t] = ma[t] + np.dot(arr.T, pcr)
    # generate the ARFIMA(p,d,q) series
    if np.isclose(d, 0):
        series = alpha + arma
    else:
        # get binomial coefficients
        bc = np.flip(_calc_arfima_binomial(d, slen + numseas))
        end = slen + numseas + 1
        series = np.zeros([slen+numseas, 1])
        for t in range (slen + numseas):
            bcr = bc[end-t-2:end]
            ars = arma[0:t+1]
            series[t] = alpha + np.dot(bcr.T, ars)
        # if negative d then single difference
        if neg:
            series1 = np.zeros([slen+numseas, 1])
            series1[0] = series[0]
            for t in range (1, slen + numseas):
                series1[t] = series[t] - series[t - 1]
            series = series1
    # trim seasoning samples and return 1d
    return series[numseas:].flatten()

# Unit tests are as follows:
#   1) generate series with ARFIMA_sim
#   2) estimate model coefficients
#   3) run ADF stationarity test
#
# Please note: there are no known ARFIMA estimation routines available
# in python open-source packages. For the three ARFIMA unit tests,
# parameter estimates were performed in R/arfima and SAS/farmafit and
# are as follows:
#   far1 = ARFIMA_sim([0.5], [-0.2, 0.2], 0.3, 1000)
#     SAS: phi = [0.50249]  theta = [-0.15113, 0.21971]  d = 0.35646
#       R: phi = [0.57048]  theta = [-0.12642, 0.23843]  d = 0.31265
#   far2 = ARFIMA_sim([0.3], [-0.4], 0.7, 1000)
#     SAS: phi = [0.25365]  theta = [-0.401356]  d = 0.75674
#       R: unable to converge
#   far3 = ARFIMA_sim([], [0.5, 0.2], -0.3, 1000)
#     SAS: phi = []  theta = [0.51174, 0.18146]  d = -0.26725
#       R: phi = []  theta = [0.50203, 0.19088]  d = -0.28356
#   far4 = ARFIMA_sim([0.2, -0.1], [], -0.7, 1000)
#     SAS: phi = [0.13011, -0.11888]  theta = []  d = -0.61225
#       R: phi = [0.18847, -0.10003]  theta = []  d = -0.67905
def main():
    # stationary white noise:
    #   wn = ARFIMA_sim([], [], 0, 1000)
    series = np.genfromtxt("results\\wn.csv", delimiter=",")
    af = ARMA(series, order=(0,0)).fit()
    assert_almost_equal(af.params[0], 0.01994, decimal=4)
    adf = adfuller(series, regression='nc')
    assert_equal(adf[2], 0)
    assert_almost_equal(adf[0], -31.16444, decimal=4)
    assert_almost_equal(adf[1], 0.00000, decimal=4)
    # stationary AR(1) with constant
    #   ar = ARFIMA_sim([0.3], [], 0, 1000, alpha=0.1)
    series = np.genfromtxt("results\\ar.csv", delimiter=",")
    af = ARMA(series, order=(1,0)).fit()
    assert_almost_equal(af.params[0], 0.11102, decimal=4)
    assert_almost_equal(af.params[1], 0.28399, decimal=4)
    adf = adfuller(series)
    assert_equal(adf[2], 2)
    assert_almost_equal(adf[0], -15.23072, decimal=4)
    assert_almost_equal(adf[1], 0.00000, decimal=4)
    # stationary MA(1) with constant
    #   ma = ARFIMA_sim([], [0.6], 0, 1000, alpha=0.25)
    series = np.genfromtxt("results\\ma.csv", delimiter=",")
    af = ARMA(series, order=(0,1)).fit()
    assert_almost_equal(af.params[0], 0.21821, decimal=4)
    assert_almost_equal(af.params[1], -0.55402, decimal=4)
    adf = adfuller(series)
    assert_equal(adf[2], 5)
    assert_almost_equal(adf[0], -17.23269, decimal=4)
    assert_almost_equal(adf[1], 0.00000, decimal=4)
    # stationary ARMA(2,1)
    #   arma = ARFIMA_sim([0.3, -0.2], [0.4], 0, 1000)
    series = np.genfromtxt("results\\arma.csv", delimiter=",")
    af = ARMA(series, order=(2,1)).fit()
    assert_almost_equal(af.params[0], -0.02283, decimal=4)
    assert_almost_equal(af.params[1], 0.29196, decimal=4)
    assert_almost_equal(af.params[2], -0.19879, decimal=4)
    assert_almost_equal(af.params[3], -0.42166, decimal=4)
    adf = adfuller(series, regression='nc')
    assert_equal(adf[2], 3)
    assert_almost_equal(adf[0], -20.00921, decimal=4)
    assert_almost_equal(adf[1], 0.00000, decimal=4)
    # non-stationary ARIMA(1,1,1)
    #   arima = ARFIMA_sim([0.3], [0.2], 1, 1000)
    series = np.genfromtxt("results\\arima.csv", delimiter=",")
    af = ARIMA(series, order=(1,1,1)).fit()
    assert_almost_equal(af.params[0], -0.00787, decimal=4)
    assert_almost_equal(af.params[1], 0.37529, decimal=4)
    assert_almost_equal(af.params[2], -0.27557, decimal=4)
    adf = adfuller(series, regression='nc')
    assert_equal(adf[2], 2)
    assert_almost_equal(adf[0], -1.95844, decimal=4)
    assert_almost_equal(adf[1], 0.04794, decimal=4)
    # stationary ARFIMA(1,0.3,2)
    #   far1 = ARFIMA_sim([0.5], [-0.2, 0.2], 0.3, 1000)
    series = np.genfromtxt("results\\far1.csv", delimiter=",")
    adf = adfuller(series, regression='nc')
    assert_equal(adf[2], 17)
    assert_almost_equal(adf[0], -3.74901, decimal=4)
    assert_almost_equal(adf[1], 0.00019, decimal=4)
    # non-stationary ARFIMA(1,0.7,1)
    #   far2 = ARFIMA_sim([0.3], [-0.4], 0.7, 1000)
    series = np.genfromtxt("results\\far2.csv", delimiter=",")
    adf = adfuller(series, regression='nc')
    assert_equal(adf[2], 9)
    assert_almost_equal(adf[0], -1.34741, decimal=4)
    assert_almost_equal(adf[1], 0.16492, decimal=4)
    # stationary ARFIMA(0,-0.3,2)
    #   far3 = ARFIMA_sim([], [0.5, 0.2], -0.3, 1000)
    series = np.genfromtxt("results\\far3.csv", delimiter=",")
    adf = adfuller(series, regression='nc')
    assert_equal(adf[2], 13)
    assert_almost_equal(adf[0], -13.15189, decimal=4)
    assert_almost_equal(adf[1], 0.00000, decimal=4)
    # stationary ARFIMA(2,-0.7,0)
    #   far4 = ARFIMA_sim([0.2, -0.1], [], -0.7, 1000)
    series = np.genfromtxt("results\\far4.csv", delimiter=",")
    adf = adfuller(series, regression='nc')
    assert_equal(adf[2], 17)
    assert_almost_equal(adf[0], -11.73815, decimal=4)
    assert_almost_equal(adf[1], 0.00000, decimal=4)

if __name__ == "__main__":
    sys.exit(int(main() or 0))