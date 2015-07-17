import numpy as np

def eig_sort(M):
    """ 
    find the sorted eigenvalue and eigenvectors
    @params M: Matrix form of the equations of motion
    @return eVals, eVecs: EigenValues and eigenVectors
    """
    eVals, eVecs = np.linalg.eig(M)
    inds = np.argsort(eVals)
    return eVals[inds], eVecs[:, inds]

def evenPoly(x, *ps):
    return sum([p * x ** (2 * n) for (n, p) in enumerate(ps)])

def oddPoly(x, *ps):
    return sum([p * x ** (2 * n + 1) for (n, p) in enumerate(ps)])

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Electrons():
    def __init__(self, xys, v0, ps, constants, **kwargs):
        """ take all input and save as default 
        @params xys: take in the x and ys in the format of array of (x, y) tuples. 
        @params kwargs.boxL: number the length of the periodicity of the boundary
        @params kwargs.cutoffL: the cutoff length of the interaction
        """
        self.oddPoly = oddPoly
        self.evenPoly = evenPoly
        if kwargs.has_key('verbose'):
            self.verbose = kwargs['verbose']
        else:
            self.verbose = False
        assert np.shape(xys)[1] == 2, 'xys should be array of (x, y) tuples'
        self.xys = xys
        self.n = np.shape(xys)[0]
        if self.verbose: print 'a total of {} electrons'.format(self.n)
        self.v0 = v0
        self.n_photons = 1
        self.ps = ps
        assert len(self.ps) > 0, 'potential parameters need to be more than 0'
        if self.verbose: print 'the potential is poly{}'.format(self.ps)


        # this is inductance, not length.
        if kwargs.has_key('boxL'): self.boxL = kwargs['boxL']
        if kwargs.has_key('cutoffL'): self.cutoffL = kwargs['cutoffL']

        self.phys = constants
        assert self.phys.has_key('m_e'), 'need to have electric constant "m_e" in constants. Or can add via self.phys.m_e'
        assert self.phys.has_key('q_e'), 'need to have electric constant "q_e" in constants. Or can add via self.phys.q_e'
        assert self.phys.has_key('k'), 'need to have electric constant "k" in constants. Or can add via self.phys.k'

        self.dx = lambda i, j: self.xys[j, 0] - self.xys[i, 0]
        if hasattr(self, 'boxL'):
            def dy (i, j):
                dyL = self.xys[i, 1] - self.xys[j, 1]
                if dyL > self.boxL/2.:
                    return dyL - self.boxL
                elif dyL < - self.boxL/2.:
                    return dyL + self.boxL
                else:
                    return dyL
            self.dy = dy
        else:
            self.dy = lambda i, j: self.xys[j, 1] - self.xys[i, 1]
        self.r = lambda i, j: 0 if i == j else ( self.dx(i, j) ** 2 + self.dy(i, j) ** 2 ) ** .5
        self.cos = lambda i, j: 0 if i == j else self.dx(i, j) / self.r(i, j)
        self.sin = lambda i, j: 0 if i == j else self.dy(i, j) / self.r(i, j)
        self.cos2theta = lambda i, j: 0 if i == j else 2 * self.cos(i, j) ** 2 - 1 #self.sin(i, j) ** 2
        self.sin2theta = lambda i, j: 0 if i == j else 2 * self.cos(i, j) * self.sin(i, j)

        ### Energy and field functions, in Joules and Micron.
        ## kij is the func for the interaction energy
        self.kij0 = self.phys['q_e']** 2 * self.phys['k'] * 1e18 # this is the spring constant
        self.cij0 = self.phys['q_e']** 2 * self.phys['k'] * 1e6 # this is for the potential energy calculation

        if self.verbose: print 'self.kij0 = {} J/m**2'.format(self.kij0)
        assert hasattr(self, 'kij0'), 'self need to have the interactive energy "kij0"'

        # k_ij is the force constant of the electron-electron interaction
        if hasattr(self, 'cutoffL'):
            def k_ij (i, j):
                r = self.r(i, j)
                if r == 0:
                    return 0
                elif r < self.cutoffL:
                    return self.kij0 / r ** 3
                elif r > self.cutoffL:
                    return 0

            self.k_ij = k_ij
        else:
            self.k_ij = lambda i, j: 0 if i == j else self.kij0 / self.r(i, j) ** 3

        ## k0 is the func for the second order dirivative of the electrical potential.
        self.ps_ddU = [(2 * n) * (2 * n - 1) * p for (n, p) in enumerate(self.ps)][1:]
        self.k0 = lambda x: self.phys['q_e'] * 1e12 * evenPoly(x, *self.ps_ddU)
        self.k_trap = lambda i: (- self.v0) * self.k0(self.xys[i, 0])

        ## E_i is the func for the electric field
        self.ps_dU = [2 * n * p for (n, p) in enumerate(self.ps)][1:]
        self.E_i = lambda i: 1e6 * self.v0 * self.phys['q_e'] * oddPoly(self.xys[i, 0], *self.ps_dU)


    def make_ee(self):
        """ make the electron sub system equations of motion
        """
        #self.meM = np.diag([self.m_e] * self.n)
        self.kM = np.array([[self.k_ij(i, j) for i in range(self.n)] for j in range(self.n)])
        self.xM = np.array([[1.5 * self.cos2theta(i, j) for i in range(self.n)] for j in range(self.n)]) * self.kM
        self.yM = np.array([[1.5 * self.sin2theta(i, j) for i in range(self.n)] for j in range(self.n)]) * self.kM
        self.xxM = 0.5 * self.kM + self.xM
        self.yyM = 0.5 * self.kM + self.yM
        self.k0M = np.diag([self.k_trap(i) for i in range(self.n)])

        ## make the electron interaction matrix (force matrix)
        self.eeM = np.zeros([2 * self.n, 2 * self.n])
        self.eeM[:self.n, :self.n] = np.diag(np.sum(self.xxM, axis=1)) - self.xxM + self.k0M
        self.eeM[:self.n, self.n:] = np.diag(np.sum(self.yM, axis=1)) - self.yM
        return self

    def make_sys(self, **kwargs):
        """ should make self.sys """
        if kwargs.has_key('omega'): self.omega = kwargs['omega']
        if kwargs.has_key('L'): self.L = kwargs['L']
        assert hasattr(self, 'omega'), 'need to have omega as an input, or object have to have .omega as an attribute'
        assert hasattr(self, 'L'), 'need to have attribute "L" in object. You can manually add this.'
        assert hasattr(self, 'eeM'), 'need to run "make_ee" to build the electron system equation \nof motion first.'

        # if kwargs.has_key('l'): self.l = kwargs['l']
        self.k_eg = self.L * self.phys['q_e'] * self.omega ** 2
        if self.verbose == True: print 'self.k_eg = {} $LCS^-2$'.format(self.k_eg)

        ## mw_norm is the normalization voltage for the microwave photon.
        if kwargs.has_key('mw_width'): self.mw_width = kwargs['mw_width']
        if kwargs.has_key('mw_potential_scale_offset'): self.mw_potential_scale_offset = kwargs['mw_potential_scale_offset']
        if kwargs.has_key('mw_lever'): self.mw_lever = kwargs['mw_lever']
        # here we use the functional form of the trapping potential to calculate the
        # electric field of the photon.
        # - `mw_potential_scale_offset` is the scale offset between the potential
        #   defined by the polynomial *ps, and the actual potental. If one is
        #   using a polynomial fit from the FEM potential, this factor should be
        #   set to 1.
        #
        # - `mw_couple_i` is the function we use to calculate the coupling with the
        #   ith electron.
        # the 1e6 is here because we are in V/micron for the potential, and here we
        # are using the electric field.
        # - the minus sign is there because the potential is upside down. X'(x) > 0 when x > 0.
        self.couple_c = self.k_eg * self.mw_potential_scale_offset * self.mw_lever * 1e6;
        self.couple_i = lambda i: self.couple_c * - oddPoly(self.xys[i, 0], *self.ps_dU)

        self.sys = np.zeros([2 * self.n + 1, 2 * self.n + 1])
        self.sys[1:, 1:] = self.eeM
        self.sys[0, 0] = self.omega ** 2 * self.L
        self.sys[0, 1:self.n + 1] = np.array([self.couple_i(i) for i in range(self.n)])
        self.sys[1:self.n + 1, 0] = np.array([self.couple_i(i) for i in range(self.n)])
        return self

    def make_eom(self):
        """ should make self.eom """
        assert hasattr(self, 'sys'), 'need to run "make_sys" to build the coupled electron-cavity system.'
        assert hasattr(self, 'n'), 'specify dimension of the system n.'
        assert hasattr(self, 'L'), 'need to have attribute "L" in object. You can manually add this.'
        assert self.phys.has_key('m_e'), 'need to have attribute "m_e" in self.phys'
        self.mM = np.diag([self.L ] + [self.phys['m_e'], ] * 2 * self.n)  # this is different from meM, which is n-tuple.
        self.eom = np.dot(np.linalg.inv(self.mM), self.sys)
        return self

    def freez_y(self, key='eom'):
        if key == 'eeM' and hasattr(self, 'eeM'):
            self.eeM[self.n:, self.n:] = np.zeros((self.n, self.n))
            self.eeM[self.n:, :self.n] = np.zeros((self.n, self.n))
            self.eeM[:self.n, self.n:] = np.zeros((self.n, self.n))
        elif key == 'eom' and hasattr(self, 'eom'):
            self.eom = self.eom[:self.n + 1, :self.n + 1]

    def solve(self, eom=None, override=False):
        if eom != None:
            return eig_sort(eom)
        if not override:
            assert np.shape(self.eom) == (2 * self.n + 1, 2 * self.n + 1)
        return eig_sort(self.eom)

    def show_xys(self, ax=None):
        if ax != None:
            ax.plot(self.xys[:, 0], self.xys[:, 1], 'o', alpha=0.5, markeredgecolor='none')
        else:
            plt.plot(self.xys[:, 0], self.xys[:, 1], 'o', alpha=0.5, markeredgecolor='none')

    def show_eeM(self, ax=None):
        if ax != None:
            p = ax.imshow(self.eeM, interpolation='none')
            cax = make_axes_locatable(ax).append_axes("right", size="10%", pad=0.05)
            plt.colorbar(p, cax=cax)
            ax.set_title('electron interaction \nmatrix', fontsize=15)
        else:
            plt.imshow(self.eeM, interpolation='none')
            plt.colorbar(shrink=0.5)
            plt.title('electron interaction matrix', fontsize=15)

    def show_sys(self, ax=None):
        if ax != None:
            p = ax.imshow(self.sys, interpolation='none')
            cax = make_axes_locatable(ax).append_axes("right", size="10%", pad=0.05)
            plt.colorbar(p, cax=cax)
            ax.set_title('cavity electron \nsystem matrix', fontsize=15)
        else:
            plt.imshow(self.sys, interpolation='none')
            plt.colorbar(shrink=0.5)
            plt.title('cavity electron \nsystem matrix', fontsize=15)

    def show_eom(self, ax=None):
        if ax != None:
            p = ax.imshow(self.eom, interpolation='none')
            cax = make_axes_locatable(ax).append_axes("right", size="10%", pad=0.05)
            plt.colorbar(p, cax=cax)
            ax.set_title('equation of motion \nnormalized by $(L, m_i)$ matrix', fontsize=15)
        else:
            plt.imshow(self.eom, interpolation='none')
            plt.colorbar(shrink=0.5)
            plt.title('equation of motion \nnormalized by $(L, m_i)$ matrix', fontsize=15)

    def test(self, *args, **kwargs):
        # plot the potential and the first order and second order of
        # the field, in the unit of V/m, and V/m^2
        if self.verbose == True:
            fig1, axes = plt.subplots(figsize=(15, 3), nrows=1, ncols=3)
            p0s = axes.flat

            xys = np.linspace(-3, 3, 51)
            ys = np.zeros(51)
            _xys = self.xys
            self.xys = np.array(zip(xys, ys))
            p0s[0].plot(xys, [self.k0(i) for i in range(len(xys))], '.-', markeredgecolor='none')
            p0s[0].set_title('potential of \nthe potential $(J)$', fontsize=15)
            p0s[1].plot(xys, [self.k1(i) for i in range(len(xys))], '.-', markeredgecolor='none')
            p0s[1].set_title('electric field $(V/m)$', fontsize=15)
            p0s[2].plot(xys, [self.k2(i) for i in range(len(xys))], '.-', markeredgecolor='none')
            p0s[2].set_title('local curvature of \nthe potential $(V/m^2)$', fontsize=15)
            # now recover the old xys
            self.xys = _xys

        fig2, axes = plt.subplots(figsize=(15, 3), nrows=1, ncols=4)
        p1s = axes.flat
        # first check the electrons
        self.show_xys(p1s[0])

        # now make the matrix
        self.make_ee()
        self.show_eeM(p1s[1])
        nanMax = np.max(np.isnan(self.eeM))
        assert nanMax == False, 'eeM has {} nan values'.format(np.sum(np.isnan(self.eeM)))

        # now check the system matrix
        self.omega = 4.95e9 * 2 * np.pi
        self.make_sys()
        self.show_sys(p1s[2])

        return self

    def show(self, *args, **kwargs):
        fig2, axes = plt.subplots(figsize=(15, 3), nrows=1, ncols=4)
        p1s = axes.flat
        # first check the electrons
        self.show_xys(p1s[0])

        # now make the matrix
        self.show_eeM(p1s[1])
        nanMax = np.max(np.isnan(self.eeM))
        assert nanMax == False, 'eeM has {} nan values'.format(np.sum(np.isnan(self.eeM)))

        # now check the system matrix
        self.show_sys(p1s[2])

        # now check the eom matrix
        self.show_eom(p1s[3])

        return self


#constants = {
#    'L': 50/(2*np.pi*f0),
#    'f0': 4.784e9, #M006T16 /#30 puffs
#    'k': 8.99e9, #J.m/C2 (F/m)
#    'q_e': 1.602e-19, #C
#    'h': 6.626e-34, #m2kg/s ~ J.s
#    'm_e': 9.1094e-31 #kg
#}
#es = Electrons(xys[0][:,1::-1], 1, ps, **constants).test().make_ee().make_sys(omega=constants['f0']*2*np.pi).make_eom()
#es.solve()   
