import numpy as np
from math import *
import sys
from matplotlib import pylab
from pylab import *
import scipy
from scipy import interpolate
import scipy.interpolate as interp

sys.path.append('/Users/luizno/Dropbox/orienta/ms/ana/writeup/gsk/cc/')

def dirctry():
    return "/Users/luizno/Dropbox/orienta/ms/ana/writeupAL/exptal/2021XSKpfig2/"

def d(lamb, n):
    '''effective bandwidth'''
    return (1.0 - 1. / lamb) / log(lamb) * lamb ** ((1. - n) / 2.)


ROUND = 1.e-18
color = ['r', 'g', 'b', 'm', 'c', 'k']


def fit(t_, wid, power):
    '''returns fit to k10'''
    x_ = np.abs(np.log(t_))
    ret = 1. / ((x_ / wid) ** power + 1.0)

    return ret * wid ** 2


def fitc(t_, wid, power):
    '''returns fit to k10'''
    x_ = (np.log(t_))
    z_ = (x_ / wid) + 1J
    y_ = -(z_ ** (-power)).real
    return y_


def alamb(lamb):
    '''returns A_\Lambda '''
    return (1. + 1. / lamb) / (1. - 1. / lamb) / 2. * log(lamb)


def calc_lamb(filename):
    '''returns lambda computed from filename'''
    ret = float(filename[8:12]) / 100.0
    return ret


def read_moments(infile, bbar0):
    '''returns moments read from infile as functions of the temperature'''
    fi = open(infile, 'r')
    lines_ = fi.readlines()
    maxlines = len(lines_)
    t_ = np.zeros(maxlines)
    l0_ = np.zeros(maxlines)
    l1_ = np.zeros(maxlines)
    l2_ = np.zeros(maxlines)
    l101_ = np.zeros(maxlines)
    l000_ = np.zeros(maxlines)
    l200_ = np.zeros(maxlines)
    count = 0
    lamb = calc_lamb(infile)
    for l, line in enumerate(lines_):
        slin_ = line.split()
        if line[0] == '\\':
            lamb = float(slin_[2])
            z = float(slin_[5])
            continue
        elif line[0:2] == '#i':
            iter = int(slin_[1])
            continue
        elif line[0] == '#':
            continue
        elif line[0] != 'A':
            bbar = float(slin_[-1])
            if (bbar >= bbar0 - ROUND) and (
                    bbar < bbar0 * sqrt(lamb) + ROUND):
                t_[count] = d(lamb, iter) / bbar
                if t_[count] < 2.e-10:
                    pass
                l0_[count] = float(slin_[1])
                l1_[count] = float(slin_[2])
                l2_[count] = float(slin_[3])
                l101_[count] = float(slin_[4])
                l000_[count] = float(slin_[5])
                l200_[count] = float(slin_[6])

                count += 1
    return ([l0_[0:count], l1_[0:count], l2_[0:count], l101_[0:count],
             l000_[0:count], l200_[0:count], t_[0:count]])


def plot_moments(filename, bbar0):
    l05__ = read_moments(filename + '.0500', bbar0)

    lamb = calc_lamb(filename)
    for i in range(3): l05__[i] = l05__[i] * (alamb(lamb)) ** 0.5
    l10__ = read_moments(filename + '.1000', bbar0)

    for l, l__ in enumerate((l05__, l10__)):
        t_ = l__[3] + ROUND
        l0_ = np.divide(l__[0], t_) * alamb(lamb) / 2.
        l1_ = np.divide(l__[1], t_) * alamb(lamb) / 2.
        l2_ = np.divide(l__[2], t_) * alamb(lamb) / 2. / pi
        semilogx(t_, l0_, color[l] + 'o')
        semilogx(t_, l1_, color[l + 2] + 'o')
        semilogx(t_, l2_, color[l + 4] + 'o')
        if l == 0:
            y0_ = l0_ / 2.
            y1_ = l1_ / 2.
            y2_ = l2_ / 2.
        else:
            y0_ += l0_ / 2.
            y1_ += l1_ / 2.
            y2_ += l2_ / 2.

    y1_ = y1_ / max(-y1_)
    semilogx(t_, y0_, color[2] + 'o')
    semilogx(t_, -y1_, color[1] + 'o')
    l0int = []
    trunc_ = []
    l1trunc_ = []
    for i, t in enumerate(t_):
        if t > 1.e-2: continue
        trunc_.append(t_)
        l1trunc_.append(l1_[i])

    for i in range(2, len(trunc_), 2):
        y_ = []
        x_ = []
        for it in range(i):
            x_.append(trunc_[it])
            y_.append(l1trunc_[it])
        print(x_, y_)
        l0int.append(scipy.integrate.simps(y_, x_, even='avg'))
    l0int_ = np.array(l0int)
    print(l0int_[0:30])

    y_ = np.diff(y0_)
    x_ = (t_[1:] + t_[:-1]) / 2.
    semilogx(x_[10:-1] / sqrt(2.), l0int_, color[0] + 'o')
    ylim(0, 1.5)
    xlim(1.e-9, 0.1)


def plot_moment(filename, bbar0, index):
    '''plots one of the moments, defined by index 0 (l0), 1(l1), 2(l2), 3(l101), or 4(l200)'''

    l05__ = read_moments(filename + '.0500', bbar0)

    lamb = calc_lamb(filename)
    for i in range(5): l05__[i] = l05__[i] * (alamb(lamb)) ** 0.5
    l10__ = read_moments(filename + '.1000', bbar0)

    for l, l__ in enumerate((l05__, l10__)):
        t_ = l__[-1] + ROUND
        l_ = np.divide(l__[index], t_) * alamb(lamb) / 2.
        if l == 0:
            y_ = l_ / 2.
        else:
            y_ += l_ / 2.

    semilogx(t_, y_, color[0] + 'o')
    xlim(1.e-10, 0.1)


def plot_moment24(filename, bbar0):
    '''plots sum of l2 and l200'''
    l05__ = read_moments(filename + '.0500', bbar0)

    lamb = calc_lamb(filename)
    for i in range(5): l05__[i] = l05__[i] * (alamb(lamb)) ** 0.5
    l10__ = read_moments(filename + '.1000', bbar0)

    for l, l__ in enumerate((l05__, l10__)):
        t_ = l__[-1] + ROUND
        if l == 0:
            l2_ = np.divide(l__[2], t_) * alamb(lamb) / 4.
            l4_ = np.divide(l__[4], t_) * alamb(lamb) / 2.
        elif l == 1:
            l2_ = l2_ + np.divide(l__[2], t_) * alamb(lamb) / 4.
            l4_ = l4_ + np.divide(l__[4], t_) * alamb(lamb) / 2.

    y_ = (l2_ + l4_) / pi ** 2 * 3.0

    semilogx(t_, y_, color[0] + 'o')
    xlim(1.e-10, 0.1)


def t0_vivaldo(lamb):
    '''returns codiagonal coefficient'''
    dlamb = 1.0 / lamb
    return (d(lamb, 1.0)
            * sqrt(1.0 - dlamb) / sqrt(1.0 - dlamb ** 3.0))


def plotMomentSingleZ(filename, fig, bbar0, index):
    '''plots one of the moments, defined by index 0 (l0), 1(l1), 2(l2), 3(l101), or
    4(l200) for single z, identified by filename, on plot fig, which
    must be defined outside routine'''

    l__ = read_moments(filename, bbar0)
    t_ = l__[-1] + ROUND
    l_ = np.divide(l__[index], t_)

    lamb = calc_lamb(filename)

    l_ = l_ / t0_vivaldo(lamb)
    length = len(l_)
    NEG_THRESHOLD = -0.01
    if l_[length / 2] < NEG_THRESHOLD:
        l_ = -1.0 * l_

    fig.semilogx(t_, l_, color[0] + 'o')
    fig.axhline(0., 1.e-9, 1., color='k')
    fig.set_xlim(min(t_), 0.1)
    fig.set_ylim(0., max(l_) * 1.1)

    return fig


def mom(filename, bbar0, index):
    '''returns one of the moments, defined by index 0 (l0), 1(l1), 2(l2), 3(l101), or
    4(l200), in filename'''

    l__ = read_moments(filename, bbar0)
    t_ = l__[-1] + ROUND
    l_ = np.divide(l__[index], t_)

    lamb = calc_lamb(filename)

    l_ = l_ / t0_vivaldo(lamb)

    return t_, l_


def plotMomentDiffSingleZ(filename, zeroGammaFile, fig, bbar0, index):
    '''plots one of the moments, defined by index 0 (l0), 1(l1), 2(l2), 3(l101), or
    4(l200) for single z, identified by filename, on plot fig, which
    must be defined outside routine; subtract zero \Gamma values'''

    l__ = read_moments(filename, bbar0)
    l0G__ = read_moments(zeroGammaFile, bbar0)
    l0_ = np.zeros_like(l__[index])  # padd l0_ with zeros to
    # match size of l_
    for il, l in enumerate(l0G__[index]):
        l0_[il] = l

    t_ = l__[-1] + ROUND
    l_ = np.divide(l__[index], t_)
    l0_ = np.divide(l0_, t_)

    l_ = l_ - l0_
    length = len(l_)
    NEG_THRESHOLD = -0.01
    if l_[length / 2] < NEG_THRESHOLD:
        l_ = -1.0 * l_

    fig.semilogx(t_, l_, color[0] + 'o')
    fig.axhline(0., 1.e-9, 1., color='k')
    fig.set_xlim(min(t_), 1.)
    fig.set_ylim(-0.1, max(l_) * 1.1)

    return fig


def showMomentSingleZ(filename, bbar0, index):
    '''returns coefficient defined by index 0 (l0), 1(l1), 2(l2),
    3(l101), or 4(l200) for single z, for all files in filename_'''

    l__ = read_moments(filename, bbar0)
    # match size of l_
    lamb = calc_lamb(filename)
    t_ = (l__[-1] + ROUND) * lamb ** 0.5
    l_ = np.divide(l__[index], t_)

    length = len(l_)
    NEG_THRESHOLD = -0.01
    if l_[length / 2] < NEG_THRESHOLD:
        l_ = -1.0 * l_

    return t_, l_


def showMomentDiffSingleZ(filename, zeroGammaFile, bbar0, index):
    '''returns coefficient defined by index 0 (l0), 1(l1), 2(l2),
    3(l101), or 4(l200) for single z, for all files in filename_'''

    l__ = read_moments(filename, bbar0)
    l0G__ = read_moments(zeroGammaFile, bbar0)
    l0_ = np.zeros_like(l__[index])  # padd l0_ with zeros to
    # match size of l_
    for il, l in enumerate(l0G__[index]):
        if il >= len(l0_): break  # in case l0G__[index] bigger than l0_
        l0_[il] = l

    lamb = calc_lamb(filename)
    t_ = (l__[-1] + ROUND) * lamb ** 0.5
    l_ = np.divide(l__[index], t_)
    l0_ = np.divide(l0_, t_)

    l_ = l_ - l0_

    length = len(l_)
    NEG_THRESHOLD = -0.01
    if l_[length / 2] < NEG_THRESHOLD:
        l_ = -1.0 * l_

    return t_, l_


def cdf0Moments(filename, bbar0, index):
    '''returns coefficient defined by index 0 (l0), 1(l1), 2(l2),
    3(l101), or 4(l200), for all files in filename_'''

    l__ = read_moments(filename, bbar0)
    # match size of l_
    lamb = calc_lamb(filename)
    t_ = (l__[-1] + ROUND) * lamb ** 0.5
    l_ = np.divide(l__[index], t_)

    length = len(l_)
    NEG_THRESHOLD = -0.01
    if l_[length / 2] < NEG_THRESHOLD:
        l_ = -1.0 * l_

    return t_, l_


def moveFiles(u, ed, gamma, lamb):
    '''moves output files from nrg run to publico directory'''
    from subprocess import call
    ROUND = 1.e-12
    lambIndex = int(100. * (lamb + ROUND))
    dataFile = "emoments%04d" % (lambIndex)
    suffix = "gam%2ded%3d" % (int((gamma + ROUND) * 10.), int((-ed + ROUND) * 1000.))

    call(['cp', '"%s.1001"%dataFile', '"../publico/%s.%s"%(dataFile, suffix)'])
    for m in range(1, 6):
        fluxi = "../publico/fluxi.%d.%04d.%s" % (m, lambIndex, suffix)
        fluxp = "../publico/fluxp.%d.%04d.%s" % (m, lambIndex, suffix)

        call(['cp', 'fluxi.%d' % m, fluxi])
        call(['cp', 'fluxp.%d' % m, fluxp])

    call(['ls', '../publico/fluxi.1.*'])
    call(['ls', '../publico/fluxi.1.*'])


def tns(lamb, n):
    '''returns matrix with Wilsons t_ns'''
    n_ = range(n)
    n_ = np.array(n_)

    ret_ = (1. - 1. / lamb) / log(lamb)
    ret_ = ret_ * (1.0 - lamb ** (-n_ - 1.0)) / (
                (1.0 - lamb ** (-2 * n_ - 1.0)) * (1.0 - lamb ** (-2 * n_ - 3.0))) ** 0.5
    ret_ = ret_ * lamb ** (-n_ / 2.0)
    return ret_


def diagHam(lamb, n, w):
    '''diagonalizes fixed-point hamiltonian for given Lambda, N, and potential w'''
    ham__ = np.zeros((n + 1, n + 1))

    t_ = tns(lamb, n + 2)
    for lin in range(n):
        ham__[lin][lin + 1] = ham__[lin + 1][lin] = t_[lin]
    ham__[n][n - 1] = t_[n - 1]
    ham__[n - 1][n] = t_[n - 1]

    ham__[0][0] = 2. * w
    ham__ = ham__ * lamb ** ((n - 1.) / 2.) * log(lamb) / (1. - 1. / lamb)

    eval_, evect__ = np.linalg.eig(ham__)

    return eval_, evect__


def erg2w(lamb, erg):
    '''finds potential scattering w that yields eigenvalue erg, just above Fermi energy,
    for odd iteration number n'''
    DELTA = 1.e-7
    TOL = 1.e-15
    n = 21
    if erg > sqrt(lamb) or erg < 0.0:
        print(f"***unexpected erg: {erg:f} should be between 0 and sqrt(lamb)")

    erg0 = sqrt((1. - lamb ** -1) * (1. - lamb ** -3))
    wtrial = log(erg / erg0) / log(lamb)
    found = False
    for loop in range(20):
        erg1 = diagHam(lamb, n, wtrial)[0][-1]
        if log(erg1 / erg) ** 2 < TOL:
            found = True
            break
        erg1p = diagHam(lamb, n, wtrial + DELTA)[0][-1]
        print("iteration %d: erg = %5.3f, w = %5.3f" % (loop, erg1, wtrial))
        wtrial += log(erg / erg1) / (log(erg1p / erg1) / DELTA)
    if found:
        delta = atan(pi * wtrial)
        print("w = %6.4f\t delta = %6.4f\t delta/pi = %6.4f" % (wtrial, delta, delta / pi))


color_ = ['orange', 'gold', 'olive', 'red']


def readFlux(filename):
    ind = open(filename)
    erg_ = []
    n_ = []
    for lin in ind:
        if lin[0] == '#' or lin[:2] == ' 0': continue
        slin = lin.split()
        n_.append(int(slin[0]))
        erg_.append(float(slin[2]))
    n_ = np.array(n_)
    erg_ = np.array(erg_)
    return n_, erg_


# from basics import Plot

def d(lamb, n):
    '''effective bandwidth'''
    return (1.0 - 1. / lamb) / log(lamb) * lamb ** ((1. - n) / 2.)


ROUND = 1.e-18
color = ['r', 'g', 'b', 'm', 'c', 'k']


def fit(t_, wid, power):
    '''returns fit to k10'''
    x_ = np.abs(np.log(t_))
    ret = 1. / ((x_ / wid) ** power + 1.0)

    return ret * wid ** 2


def fitc(t_, wid, power):
    '''returns fit to k10'''
    x_ = (np.log(t_))
    z_ = (x_ / wid) + 1J
    y_ = -(z_ ** (-power)).real
    return y_


def alamb(lamb):
    '''returns A_\Lambda '''
    return (1. + 1. / lamb) / (1. - 1. / lamb) / 2. * log(lamb)


def calc_lamb(filename):
    '''returns lambda computed from filename'''
    ret = float(filename[8:12]) / 100.0
    return ret


def read_moments(infile, bbar0):
    '''returns moments read from infile as functions of the temperature'''
    fi = open(infile, 'r')
    lines_ = fi.readlines()
    maxlines = len(lines_)
    t_ = np.zeros(maxlines)
    l0_ = np.zeros(maxlines)
    l1_ = np.zeros(maxlines)
    l2_ = np.zeros(maxlines)
    l101_ = np.zeros(maxlines)
    l000_ = np.zeros(maxlines)
    l200_ = np.zeros(maxlines)
    count = 0
    lamb = calc_lamb(infile)
    for l, line in enumerate(lines_):
        slin_ = line.split()
        if line[0] == '\\':
            lamb = float(slin_[2])
            z = float(slin_[5])
            continue
        elif line[0:2] == '#i':
            iter = int(slin_[1])
            continue
        elif line[0] == '#':
            continue
        elif line[0] != 'A':
            bbar = float(slin_[-1])
            if (bbar >= bbar0 - ROUND) and (
                    bbar < bbar0 * sqrt(lamb) + ROUND):
                t_[count] = d(lamb, iter) / bbar
                if t_[count] < 2.e-10:
                    pass
                l0_[count] = float(slin_[1])
                l1_[count] = float(slin_[2])
                l2_[count] = float(slin_[3])
                l101_[count] = float(slin_[4])
                l000_[count] = float(slin_[5])
                l200_[count] = float(slin_[6])

                count += 1
    return ([l0_[0:count], l1_[0:count], l2_[0:count], l101_[0:count],
             l000_[0:count], l200_[0:count], t_[0:count]])


def t0_vivaldo(lamb):
    '''returns codiagonal coefficient'''
    dlamb = 1.0 / lamb
    return (d(lamb, 1.0)
            * sqrt(1.0 - dlamb) / sqrt(1.0 - dlamb ** 3.0))


def plotMomentSingleZ(filename, fig, bbar0, index):
    '''plots one of the moments, defined by index 0 (l0), 1(l1), 2(l2), 3(l101), or
    4(l200) for single z, identified by filename, on plot fig, which
    must be defined outside routine'''

    l__ = read_moments(filename, bbar0)
    t_ = l__[-1] + ROUND
    l_ = np.divide(l__[index], t_)

    lamb = calc_lamb(filename)

    l_ = l_ / t0_vivaldo(lamb)
    length = len(l_)
    NEG_THRESHOLD = -0.01
    if l_[length / 2] < NEG_THRESHOLD:
        l_ = -1.0 * l_

    fig.semilogx(t_, l_, color[0] + 'o')
    fig.axhline(0., 1.e-9, 1., color='k')
    fig.set_xlim(min(t_), 0.1)
    fig.set_ylim(0., max(l_) * 1.1)

    return fig


def mom(filename, bbar0, index):
    '''returns one of the moments, defined by index 0 (l0), 1(l1), 2(l2), 3(l101), or
    4(l200), in filename'''

    l__ = read_moments(filename, bbar0)
    t_ = l__[-1] + ROUND
    l_ = np.divide(l__[index], t_)

    lamb = calc_lamb(filename)

    l_ = l_ / t0_vivaldo(lamb)

    return t_, l_


import subprocess as sbpr


def set(filename):
    '''computes conductance'''
    bbmin = 0.275

    bbmin -= 1.e-5
    bbmax = 2. * bbmin
    bbmax += 1.e-5

    t_ = []
    g_ = []
    gset__ = np.loadtxt(filename, usecols=(0, 1, 7))
    for lin, line_ in enumerate(gset__):
        if line_[0] == '#': continue
        t = float(line_[0])
        g = float(line_[1]) / t
        bbar = float(line_[2])
        if (bbar > bbmin) & (bbar < bbmax):
            t_.append(t)
            g_.append(g)
    return t_, g_


def scd(filename, moment):
    '''computes conductance for side-coupled device'''
    column_ = {"temp": 0, "l^0_11": 1, "l^1_11": 2, "l^2_11": 3,
               "l^1_01": 4, "l^0_00": 5, "l^2_00": 6, "beta_bar": 7}
    col = column_[moment]
    print(col)
    bbmin = 0.275

    bbmin -= 1.e-5
    bbmax = 2. * bbmin
    bbmax += 1.e-5

    t_ = []
    y_ = []
    col_t = column_["temp"]
    col_bbar = column_["beta_bar"]
    try:
        yscd__ = np.loadtxt(filename,
                            usecols=(col_t, col, col_bbar)
                            )
    except:
        print(f"{filename, col_t, col, col_bbar}")
        exit(1)
    for lin, line_ in enumerate(yscd__):
        if line_[0] == '#': continue
        t = float(line_[0])
        y = float(line_[1]) / t
        bbar = float(line_[2])
        if (bbar > bbmin) & (bbar < bbmax):
            t_.append(t)
            y_.append(y)
    return t_, y_


def average(moment, want_set=False):
    """returns temperatures and moment, from following list:
    "l_0", "l_1", "l_2", "l_01_1", "l_00_0", "l_00_1", "l_00_2"
    """
    if want_set == 1:
        t05_, g05_ = set('emoments0400.0500')
        t10_, g10_ = set('emoments0400.1000')
        ud = open('gset.txt', 'w')
    else:
        t05_, y05_ = scd(directory() + 'sandbox/emoments0400.0500', moment)
        t10_, y10_ = scd(directory() + 'sandbox/emoments0400.1000', moment)
        udname = f"{moment:}_scd.txt"
        ud = open(udname, 'w')

    y05_ = np.array(y05_)
    y10_ = np.array(y10_)

    y_ = 0.25 * (y05_ + y10_)
    t_ = np.array(t05_)

    for it, t in enumerate(t_):
        y = y_[it]
        ud.write(f"{t:g}\t\t{y:g}\n")

    return t_, np.array(y_)

def average_tchi(bbar_):
    """ reads Tchi data from files tchi.s.050 and tchi.s.100 and returns average"""
    bbmin = bbar_[0]
    bbmax = bbar_[1]
    print(bbmin, bbmax)
    tchi050__ = np.loadtxt(directory() + "sandbox/tchi.s.050",
                         usecols = (0, 1, 5)) # T, Tchi, bbar
    tchi050__ = tchi050__.transpose()
    tchi100__ = np.loadtxt(directory() + "sandbox/tchi.s.100",
                           usecols = (0, 1, 5))
    tchi100__ = tchi100__.transpose()

    t50_ = []
    tchi50_ = []
    for it, t in enumerate(tchi050__[0]):
        bbar = tchi050__[2][it]
        if bbar < bbmin or bbar > bbmax: continue
        t50_.append(t)
        tchi50_.append(tchi050__[1][it])
    t50_ = np.array(t50_)
    tchi50_ = np.array(tchi50_)

    t100_ = []
    tchi100_ = []
    for it, t in enumerate(tchi100__[0]):
        bbar = tchi100__[2][it]
        if bbar < bbar_[0] or bbar > bbar_[1]: continue
        t100_.append(t)
        tchi100_.append(tchi100__[1][it])
    t100_ = np.array(t100_)
    tchi100_ = np.array(tchi100_)

    t_ = t50_
    tchi_ = 0.5 * (tchi50_ + tchi100_)

    return t_, tchi_

def prep(u, gam, w, vg, iz):
    '''prepares sik.dat to be run with given z, defined by integer iz==10*z'''

    ind = np.loadtxt(directory() + "gsk.dat", dtype='str')
    ud = open(directory() + "sik.dat", 'w')
    for n, lin in enumerate(ind):
        if lin[1] == 'U':
            ud.write('{:<7.4f}\t\tU\n'.format(u))
        elif lin[1] == 'Gamma':
            ud.write('{:<7.4f}\t\tGamma\n'.format(gam))
        elif lin[1] == 'W':
            ud.write('{:<7.4f}\t\tW\n'.format(w))
        elif lin[1] == 'Ed':
            ud.write('{:<7.4f}\t\tEd\n'.format(vg))
        elif lin[1] == 'z':
            if iz == 5:
                print('z=0.5')
                ud.write('0.5\t\tz\n')
            elif iz == 10:
                print('z=1.0')
                ud.write('1.0\t\tz\n')
            else:
                print('***unexpected iz in prep')
        else:
            ud.write('{:}\t\t{:}\n'.format(lin[0], lin[1]))


def gsk(u, gam, w, vg):
    '''run gsk for given vg and z=0.5, z=1'''
    for iz in [5, 10]:
        prep(u, gam, w, vg, iz)
        cmd = 'cd '+directory(); './gsk'
        sbpr.call(cmd, shell=True,stdout=sbpr.PIPE)


def nrgSpan(u, gam, w):
    '''runs nrg computation of T=0 conductance from vg=-2u to u'''
    suffix = 'u{:4.2f}gam{:5.3f}w{:4.2f}'.format(u, gam, w)
    filename = 'nrgSET{:s}.dat'.format(suffix)
    fset = open(filename, 'w')
    filename = 'nrgSCD{:s}.dat'.format(suffix)
    fscd = open(filename, 'w')
    for vg in np.linspace(-2 * u, u, 61):
        gsk(u, gam, w, vg, fset, fscd)
    fset.close()
    fscd.close()

def save2file(t, moment, filename):
    """save vectors t and moment to filename"""
    ud = open(directory()+"sandbox/"+filename+".txt", "w")
    for it, t in enumerate(t):
        ud.write(f"{t:11.4g}\t\t{moment[it]:11.4g}\n")

    ud.close()


def read_from_file(filename):
    """ reads columns 1 and 2 from filename"""
    ind__ = np.loadtxt(directory()+"sandbox/" + filename+'.txt',
                       usecols=(0,1))
    x_ = np.zeros([len(ind__)])
    y_ = np.zeros_like(x_)

    for lin, line_ in enumerate(ind__):
        x_[lin] = line_[0]
        y_[lin] = line_[1]

    return x_, y_

def divide(x_, xn, num, xd, denom):
    """ calculates the ratio numerator/denominator
    at the abscissae x_"""

    numerator = interp.InterpolatedUnivariateSpline(xn, num)
    denominator = interp.InterpolatedUnivariateSpline(xd, denom)

    return x_, numerator(x_) / denominator(x_)

def thermal_sequence():
    """returns thermal sequence ranging from 1e-5 to 1e5 in log10 steps of 0.01"""
    lt_ = np.arange(-5, 5, 0.01)
    t_ = np.power(10., lt_)
    return t_

def univ_power(exp_filename, tk):
    ''' reads thermopower data from exp_filename and returns
    l_0_univ, l_1_univ(t/T_k)/s(t/T_k)'''

    t_ = thermal_sequence()
    lt_ = np.log(t_) / np.log(10.)

    name = exp_filename
    texp_, sexp_ = read_from_file(name)
    name = 'l0univ'
    t0univ_ , l0univ_ = read_from_file(name)
    name = 'l1univ'
    t1univ_, l1univ_ = read_from_file(name)
    ltx = np.log(texp_/tk)/np.log(10.)
    s = interp.InterpolatedUnivariateSpline(ltx, sexp_)

    lt0 = np.log(t0univ_)/np.log(10)
    l0 = interp.InterpolatedUnivariateSpline(lt0, l0univ_)

    lt1 = np.log(t1univ_)/np.log(10.)
    l1 = interp.InterpolatedUnivariateSpline(lt1, l1univ_)

    y_ = l1(lt_) / s(lt_)
    x_ = l0(lt_)

    return x_, y_, lt_, s(lt_), l1(lt_)

def read_from_universal(filename):
    """" returns temperatures and transport property from filename"""
    ind__ = np.loadtxt(directory()+"universal/" + filename+'.txt',
                       usecols=(0,1))
    x_ = np.zeros([len(ind__)])
    y_ = np.zeros_like(x_)

    for lin, line_ in enumerate(ind__):
        x_[lin] = line_[0]
        y_[lin] = line_[1]

    return x_, y_

def l0(del_pi, tk):
    """given phase shift delta and Kondo temperature tk,
    returns L_00^0 moment as a function of T"""
    delta = del_pi * pi
    t, l0u = read_from_universal("l0univ")
    l0_ = -cos(2.*delta) * l0u + 0.5* (1.0 + cos(2. * delta))
    t_ = t * tk

    return t_, l0_


def l1(del_pi, tk):
    """given phase shift delta and Kondo temperature tk,
    returns L_00^1 moment as a function of T"""
    delta = del_pi * pi
    t, l1u = read_from_universal("l1univ")
    l1_ = sin(2.*delta) * l1u
    t_ = t * tk

    return t_, l1_


def l2(del_pi, tk):
    """given phase shift delta and Kondo temperature tk,
    returns L_00^2 moment as a function of T"""
    delta = del_pi * pi
    t, l2u = read_from_universal("l2univ")
    pi2_3 = np.power(pi, 2.) / 3.
    l2_ = -cos(2.*delta) * l2u + pi2_3 * 0.5 * (1.0 + cos(2. * delta))
    t_ = t * tk

    return t_, l2_

def transport_moments(del_pi, tk):
    """ returns temperature and three transport moments for
    given delta/pi and T_K"""
    t_, l0_ = l0(del_pi, tk)
    t_, l1_ = l1(del_pi, tk)
    t_, l2_ = l2(del_pi, tk)

    return t_, l0_, l1_, l2_

def kb_over_e():
    """returns Boltzmann's constant divided by electron charge, in \mu V / K"""
    return (1.380649e-23 / 1.602177e-19) * 1.0e6

def wFranz(del_pi, tk):
    """ returns temperatures and Wiedemann-Franz ratio for
    given delta/pi and T_K """

    t_, l0_ = l0(del_pi, tk)
    t_, l1_ = l1(del_pi, tk)
    t_, l2_ = l2(del_pi, tk)

    return t_, l2_/l0_ - np.power(l1_/l0_, 2.)

def s_expt(t_):
    """Returns experimental data from Bashir et al. JP:Conf Series 592, 012004 (2015)
    """

    # fitting parameters
    J = 2.5
    Ef = 10.7 # 17.3
    Wf = 96 * pi / (2*J+1.0) # 94
    pi2_3 = np.power(pi,2.0) / 3.0
    ret_ = 2. * pi2_3 * Ef * t_
    ret_ /= (pi2_3 * np.power(t_, 2.0)
             + np.power(Ef, 2.0)
             + np.power(Wf, 2.0)
             )

    return t_, ret_

def linear_regress(x_, y_):
    """returns m, c coefficients of straight line fitting x to y"""
    A = np.vstack([x_, np.ones(len(x_))]).T
    m, c = np.linalg.lstsq(A, y_, rcond=None)[0]
    return m, c


def fit_thermopower(t_, s_, del_pi):
    """ returns linear coefficients m and c resulting from linear regression applied to
    L_0^univ X L_1^univ/s_ mapping. Assume t_ sequence from thermal_sequence() """

    tu, l0u = read_from_universal("l0univ")
    tu, l1u = read_from_universal("l1univ")
    ty, l0y, l1y, l2y = transport_moments(del_pi, 1.)

    arg_max_s = np.argmax(s_)
    arg_max_r = np.argmax(l1y/l0y)

    n_s = arg_max_s + 30
    n_r = arg_max_r + 30
    n0_s = n_s - n_r # vectors s and r must have same size
    # trtrunc = tu[0:n_r]
    tstrunc = t_[n0_s:n_s]

    ud = open(directory() + "sandbox/linear_map.txt", "w")
    x_ = np.zeros([n_r])
    y_ = np.zeros_like(x_)
    for ll in range(n_r):
        ratio = l1u[ll] / s_[n0_s+ll]
        x_[ll] = l0u[ll]
        y_[ll] = ratio

        ud.write(f"{x_[ll]:11.4g}\t{y_[ll]:11.4g}\n")
    ud.close()
    m, c = linear_regress(x_, y_)
    return m, c, tstrunc, x_, y_

def fit_thermopower_tk(t_expt, s_expt, tk):
    """ returns linear coefficients m and c resulting from linear regression applied to
    L_0^univ X L_1^univ/s_ mapping. Assumes t_expt forms log sequence """

    tu, l0u = read_from_universal("l0univ")
    tu, l1u = read_from_universal("l1univ")

    l0 = interp.InterpolatedUnivariateSpline(tu, l0u)
    l1 = interp.InterpolatedUnivariateSpline(tu, l1u)

    l0_t = l0(t_expt / tk)
    l1_t = l1(t_expt / tk)

    nx = len(t_expt)
    ud = open(directory() + "sandbox/linear_map.txt", "w")
    x_ = np.zeros(nx)
    y_ = np.zeros_like(x_)
    for ll in range(nx):
        ratio = l1_t[ll] / s_expt[ll]
        x_[ll] = l0_t[ll]
        y_[ll] = ratio

        ud.write(f"{x_[ll]:11.4g}\t{y_[ll]:11.4g}\n")
    ud.close()
    m, c = linear_regress(x_, y_)
    return m, c, t_expt / tk, x_, y_

def fit_thermopower_tk_linear_t(t_expt, s_expt, tk):
    """ returns linear coefficients m and c resulting from linear regression applied to
    L_0^univ X L_1^univ/s_ mapping. Assumes t_expt in linear sequence  """

    tu, l0u = read_from_universal("l0univ")
    tu, l1u = read_from_universal("l1univ")

    l0_log = interp.InterpolatedUnivariateSpline(log(tu), l0u)
    l1_log = interp.InterpolatedUnivariateSpline(log(tu), l1u)

    l0_t = l0_log(np.log(t_expt / tk))
    l1_t = l1_log(np.log(t_expt / tk))

    nx = len(t_expt)
    ud = open(directory() + "sandbox/linear_map.txt", "w")
    x_ = np.zeros(nx)
    y_ = np.zeros_like(x_)
    for ll in range(nx):
        ratio = l1_t[ll] / s_expt[ll]
        x_[ll] = l0_t[ll]
        y_[ll] = ratio

        ud.write(f"{x_[ll]:11.4g}\t{y_[ll]:11.4g}\n")
    ud.close()
    m, c = linear_regress(x_, y_)
    return m, c, t_expt / tk, x_, y_

def exptal_yb(tk):
    """"""
    # read experimental data
    dir_Kohler = "/Users/luizno/Dropbox/orienta/ms/ana/writeupAL/exptal/dados/UKohler/"

    rho__ = np.loadtxt(dir_Kohler +"rho_mag.txt", usecols=(0,1))
    t_rho_ = np.zeros([len(rho__)])
    rho_ = np.zeros_like(t_rho_)

    for lin, line_ in enumerate(rho__):
        t_rho_[lin] = line_[0]
        rho_[lin] = line_[1]

    thermo__ = np.loadtxt(dir_Kohler +"thermopower_Yb_x01.txt", usecols=(0,1))
    t_thermo_ = np.zeros([len(thermo__)])
    thermo_ = np.zeros_like(t_thermo_)
    for lin, line_ in enumerate(thermo__):
        t_thermo_[lin] = line_[0]
        thermo_[lin] = line_[1]

    return t_rho_, rho_, t_thermo_, thermo_

def interp_rho_thermo(tt, t_rho_, rho_, t_thermo_, thermo_):
    """ given temperature T, returns rho(T) and thermo(T)
     input: tt == T
            t_rho_, rho_ = experimental data (from exptal_yb())
            t_thermo_, thermo_ = experimental data (from exptal_yb())
     output: rho(T), thermopower(T) """

    log_t_rho_ = np.log(t_rho_) # resistivity data logarithmically sampled; thermopower, linearly
    rho_log = interp.InterpolatedUnivariateSpline(log_t_rho_, rho_)
    s_ = interp.InterpolatedUnivariateSpline(t_thermo_, thermo_)

    logt = log(tt)
    return rho_log(logt), s_(tt)

def plot_rho_thermo():
    """ plots energy moment l1, computed from experimental data as a function of L_01^(1)
    exptal: L_1 = S(T) * L_0
            L_0 = G(T) / e^2
    """
#    plot(log_t_rho_, rho_log(log_t_rho_))
#    plot(log_t_rho_, rho_, 'o')

    # read universal moment
    t_univ_ , l1_univ_ = read_from_universal("l1univ")

    # limits of abscissa
    # t_min = t_thermo_[0] / tk
    # t_max = t_thermo_[-1] / tk

    # t_ = []
    # it_ = [] # index to wanted temperature entry
    # for it, t in enumerate(t_univ_):
        #if t > t_max:
        #    break
        #if t < t_min:
        #    continue
        #t_.append(t)
        #it_.append(it)

    # dim = len(t_)
    # x_ = np.zeros_like(t_)
    # y_ = np.zeros_like(t_)
    # for count in range(dim):
      #  it = it_[count]
       # x_[count] = l1_univ_[it]
       # log_temp = log(t_[count]/tk)
       # y_[count] = rho_log(log_temp) * s_log(log_temp)

#    plot (x_, y_, 'o')

    #return t_, x_, y_


def directory():
    """ returns cc directory"""
    return "/Users/luizno/Dropbox/orienta/ms/ana/writeupAL/gsk/cc/enrg/"

def jk(u, gam, vg):
    """returns Schrieffer-Wolff constants J and K, from 1980KWW1044 Eq. (3.38)"""
    j = 2 * gam / pi / abs(vg) + 2 * gam / pi / (u + vg)
    k = gam / 2. / pi / abs(vg) - gam / 2. / pi / (u + vg)

    return j, k

def guniv(filename, tk):
    """reads data from file guniv.txt and gXt data from file filename and returns
    g(t/tk) vs guniv(t/tk)
    input:
    filename (g vs. T data)
    tk = T_K
    output:
     g(T/T_K)
     guniv(T/T_K)
     """
    get__ = np.loadtxt(dirctry() + "x=0-0.87/"+ filename)
    texp_ =[]
    gexp_ =[]

    for count, linexp_ in enumerate(get__):
        texp_.append(np.log(linexp_[0] / tk))
        gexp_.append(1./linexp_[1])  # want conductance, but exptal file contains resistivity
    print(f"[{texp_[0]:},{texp_[-1]}]")

    guni_ = []
    tuni_ = []
    gut__ = np.loadtxt(dirctry() + "guniv.txt")
    for count, linuv_ in enumerate(gut__):
        temp_univ = linuv_[0] #np.log(10.) * linuv_[0]  # tuniv recorded as log10(t/tk)
        g_univ = -linuv_[1] / 10. + 0.5  # guniv recorded 10. * (g(t/tk) - 0.5)
#        if temp_univ < texp_[0] or temp_univ > texp_[-1]:
#            continue
#        print(temp_univ, g_univ)
        tuni_.append(temp_univ)
        guni_.append(g_univ)

    tuni_ = np.array(tuni_)
    guni_ = np.array(guni_)
    texp_ = np.array(texp_)
    gexp_ = np.array(gexp_)

    guni_interp_ = np.interp(texp_, tuni_, guni_)

    return guni_interp_, gexp_, texp_

def find_fit(filename, tk, alpha, beta, lims_):
    """ plots linear fit from filename data
    input:
    filename == data file
    tk == T_K
    alpha = trial slope
    beta = trial intercept
    lims_ == [xmin, xmax, ymin, ymax] == plot limits
    """
    guniv_, gexp_, texp_ = guniv(filename, tk)

    plot(guniv_, gexp_, color='orange', marker='.')
    xmin = lims_[0]
    xmax = lims_[1]
    ymin = lims_[2]
    ymax = lims_[3]
    xlim(xmin , xmax)
    ylim(ymin, ymax)

    x_ = np.linspace(xmin, xmax)
    y_ = alpha * x_ + beta
    plot(x_, y_, color = 'blue')
    c_2delta = -1./(1.+2*beta/alpha)
    # linear fit to -2cos(2\delta)*G_univ + (1+\cos(2\delta)
    if abs(c_2delta) <= 1.0:
        print(f"delta/pi = {acos(c_2delta)/2.0/pi:5.3f}")
    else:
        print(f"*** => cos(delta) = {c_2delta:5.3f}")

    in_dat_ = [tk, alpha, beta]
    g_dat_ = [texp_, gexp_, guniv_]
    return in_dat_, g_dat_

def trial_fit(filename, tk, delta_pi, g0):
    """ plots linear fit from filename data
    uses g_0 = 0.03147 (from exptal data)
    input:
    filename == data file (without extension)
    tk == T_K
    delta_pi = trial ratio delta/pi
    lims_ == [xmin, xmax, ymin, ymax] == plot limits
    """
    guniv_, gexp_, texp_ = guniv(filename + '.txt', tk)
    norm_g_exp_ = gexp_ / 2.0 / g0
    cla()
    plot(guniv_, norm_g_exp_, color='orange', marker='.')

    ud = open(dirctry() + "g_vs_guniv.try", "w")
    ud.write(f"g_universal\t g_exp\n")
    for ct, guni in enumerate(guniv_):
        ud.write(f"{guni:8.4f}\t{norm_g_exp_[ct]:8.4f}\n")
    ud.close()


    xmin = 0.
    xmax = 1.
    ymin = 0.
    ymax = 1.
    xlim(xmin , xmax)
    ylim(ymin, ymax)

    alpha = -2.0* cos(2.0 * pi* delta_pi)
    beta = 1.0 + cos(2.0 * pi * delta_pi)

    x_ = np.linspace(xmin, xmax)
    y_ = (alpha * x_ + beta) / 2.0
    plot(x_, y_, color = 'blue')
    c_2delta = -1./(1.+2*beta/alpha)
    # linear fit to -2cos(2\delta)*G_univ + (1+\cos(2\delta)

    in_dat_ = [tk, delta_pi, g0]
    g_dat_ = [texp_, gexp_, g0 * (alpha * guniv_ + beta)]
    return in_dat_, g_dat_

def plot_resist(filename, tk, delta_pi, g0):
    """ compares universal resistivity as a function of T with exptal data
    input:
    tk == T_K
    delta_pi == delta/pi
    g0 == G_0 (appr. 0.0315)
    output:
    t_exptal_ == experimental temperatures (K)
    g_exp_ == experimental conductances
    g_fit_ == fit, returned by trial fit
    """
    maparams_, g_dat_ = trial_fit(filename, tk, delta_pi, g0)
    texp_ = g_dat_[0]
    g_exp_ = g_dat_[1]
    t_exptal_ = tk * np.exp(texp_)
    cla()
    xlim(0.5, 300)
    ylim(0, 40)
    semilogx(t_exptal_, 1. / g_exp_, '.', color='orange')
    g_fit_ = g_dat_[2]
    semilogx(t_exptal_, 1. / g_fit_, 'blue',linewidth=2 )
    return t_exptal_, g_exp_, g_fit_


