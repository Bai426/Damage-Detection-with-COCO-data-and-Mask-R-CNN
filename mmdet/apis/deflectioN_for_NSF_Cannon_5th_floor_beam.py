import pandas as pd
ma = pd.read_csv('/media/bys2058/Elements/NSF project data/results/Lk_5th_top_beam_motion_affine_stablize_5_1_22-12.csv', header=None)
# print(a)    ## score_thr = 0.8
### it shows that lower threshold of score for mask will generate different number of bbox!!!
mb = pd.read_csv('/media/bys2058/Elements/NSF project data/results/Lk_5th_top_beam_motion_affine_stablize_5_1_22-13.csv', header=None)
# print(ma)

import matplotlib.pyplot as plt
import datetime
import numpy as np

i_motion5 = []
for i in range(len(ma)):
    i_motion5.append(i)
len(i_motion5)

i_motion6 = []
for i in range(len(ma)):
    i_motion6.append(i)
len(i_motion6)

i_5 = np.array(i_motion5)/15
x_lk = np.array(ma[3]-np.mean(ma[3][0:10]))   ### use the average of first 10 measurement as the initial measurement
y_lk = np.array(ma[4]-np.mean(ma[4][0:10]))

i_6 = np.array(i_motion6)/15
x_lkm = np.array(mb[3]-np.mean(mb[3][0:10]))
y_lkm = np.array(mb[4]-np.mean(mb[4][0:10]))

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])     ## y must be numpy array
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

ysg = savitzky_golay(y_lk, window_size=31, order=2)
ysgm = savitzky_golay(y_lkm, window_size=31, order=2)
# plt.figure(figsize=(15,5), dpi=500)
# plt.plot(i_5, x_lk, color='tab:purple')
# plt.gca().set(title='Horizontal displacement of loading cab', xlabel='frames', ylabel='Displacement (pixel)')
# plt.show()
# plt.figure(figsize=(20,5), dpi=500)
# plt.plot(time, x, color='tab:purple')
# plt.gca().set(title='Horizontal Displacement of box', xlabel='time (s)', ylabel='Displacement (pixel)')
# plt.show()
#
# plt.figure(figsize=(15,5), dpi=500)
# plt.plot(i_5, y_lk, color='tab:purple')
# plt.gca().set(title='Vertical displacement of loading cab', xlabel='frames', ylabel='Displacement (pixel)')
# plt.show()
plt.figure(figsize=(20,5), dpi=100)
# plt.plot(i_1, y_d*.333, 'b', linewidth=1.5, label='Measurement from the camera with Mask R-CNN + SIFT')
plt.plot(i_5, y_lk*.1, 'm', linewidth=1, label='Measurement from the camera with Lucas Kanade tracker')
plt.plot(i_5, ysg*.1, 'b', linewidth=2, label='Filtered measurement from LK tracker')
plt.plot(i_6, y_lkm*.12, 'g', linewidth=1, label='Measurement from the camera with Lucas Kanade tracker')
plt.plot(i_6, ysgm*.12, 'r', linewidth=2, label='Filtered measurement from LK tracker')
# plt.plot(i_1, deflection, 'r',linewidth=2, label='Measurement from dial gauge (true measurement)')
plt.xlabel('time (s)')
plt.ylabel('Displacement (inch)')
plt.grid()
plt.legend()
plt.show()