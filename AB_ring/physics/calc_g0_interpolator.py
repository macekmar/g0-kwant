from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
import scipy as sc
import numpy as np
import kwant
import adaptive

class fun_piece():
    def __init__(self, k, engs, nb_sites, nb_leads, nb_channels, interpolator=sc.interpolate.Akima1DInterpolator):
        self.k = k
        self.engs = engs
        self.nb_sites = nb_sites
        self.nb_leads = nb_leads
        self.nb_channels = nb_channels
        self.interpolator = interpolator
        self.wf_val = -1
        self.wf_interp = None
        
    def get_data(self, syst, world):
        # world = MPI.COMM_WORLD
        wf_val = np.zeros((len(self.engs), self.nb_leads, self.nb_channels, self.nb_sites), dtype=np.complex)
        for i, e in enumerate(self.engs):
            if i % world.size == world.rank:
                wf = kwant.wave_function(syst, energy=e)
                wf_val[i, ...] = np.array([wf(j) for j in range(self.nb_leads)])
        self.wv_val = world.allreduce(wf_val)
        print(self.wv_val)

    def get_interpolators(self):
        self.wf_interp = [[[[None,None] for site in range(self.nb_sites)] for channel in range(self.nb_channels)] for lead in range(self.nb_leads)]
        for lead in range(self.nb_leads):
            for channel in range(self.nb_channels):  # caveat: at different energies a different number of channels can be opened!
                for site in range(self.nb_sites):
                    self.wf_interp[lead][channel][site][0] = self.interpolator(self.k, self.wf_val[:, lead, channel, site].real)
                    self.wf_interp[lead][channel][site][1] = self.interpolator(self.k, self.wf_val[:, lead, channel, site].imag)

    def __call__(self, k, lead, channel, site):
        return self.wf_interp[lead][channel][site][0](k) + 1j*self.wf_interp[lead][channel][site][1](k)

class wave_fun():
    def __init__(self, syst, nb_pts, eps, k_cut=[], gamma_wire=1.0):
        self.syst = syst
        self.nb_pts = nb_pts
        self.eps = eps
        self.gamma_wire = gamma_wire
        self.nb_sites = self.syst.hamiltonian_submatrix().shape[0]
        self.nb_leads = 2
        self.nb_channels = 1

        self.k_int = []
        self.pieces = []
        k_int = [[0, self.eps]] + k_cut + [[np.pi, self.eps]]
        for i in range(1,len(k_int)):
            k_lim = [k_int[i-1][0] + k_int[i-1][1], k_int[i][0] - k_int[i][1]]
            k_vec = np.linspace(k_lim[0], k_lim[1], self.nb_pts)
            engs = 2*np.abs(self.gamma_wire)*np.cos(k_vec)
            self.k_int.append(k_lim)
            self.pieces.append(fun_piece(k_vec, engs, self.nb_sites, self.nb_leads, self.nb_channels))
        for (k, eps_k) in k_cut:
            k_lim = [k - eps_k, k + eps_k]
            k_vec = np.linspace(k_lim[0], k_lim[1], 10)
            engs = 2*np.abs(self.gamma_wire)*np.cos(k_vec)
            self.k_int.append(k_lim)
            self.pieces.append(fun_piece(k_vec, engs, self.nb_sites, self.nb_leads, self.nb_channels, interpolator=sc.interpolate.interp1d))
        
    def get_data(self, world):
        for i in range(len(self.pieces)):
            self.pieces[i].get_data(self.syst, world)
    
    def get_interpolators(self):
        for i in range(len(self.pieces)):
            self.pieces[i].get_interpolators()


    def __call__(self, k, lead, channel, site):
        if np.isscalar(k):
            i = np.argmax([(k > k_[0]) and (k < k_[-1]) for k_ in self.k_int])
            return self.pieces[i](k, lead, channel, site)
        else:
            test = np.vstack([np.logical_and((k > k_[0]),  (k < k_[-1])) for k_ in self.k_int]).T
            indices = np.argmax(test, axis=1)
            res = np.zeros((len(k)),dtype=np.complex)
            for i in range(len(k)) :
                res[i] = self.pieces[indices[i]](k[i], lead, channel, site)
            return res


class wave_fun_adapt():
    def __init__(self, syst, nb_pts, eps, gamma_wire=1.0):
        self.syst = syst
        self.nb_pts = nb_pts
        self.eps = eps
        self.gamma_wire = gamma_wire
        self.nb_sites = self.syst.hamiltonian_submatrix().shape[0]
        self.nb_leads = 2
        self.nb_channels = 1
        self.k_vec = None
        self.wf_val = None
        self.wf_interp = None

    def wf_fun(self, k):
        e = 2*np.abs(self.gamma_wire)*np.cos(k)
        wf = kwant.wave_function(self.syst, energy=e)
        res = np.array([[wf(i).real, wf(i).imag] for i in range(self.nb_leads)])
        return np.moveaxis(res, 1, -1) #re, im should be last axis

    def get_data(self, mpi=False):
        learner = adaptive.Learner1D(self.wf_fun, bounds=[self.eps, np.pi - self.eps])
        # if mpi:
        #     runner = adaptive.BlockingRunner(learner, goal=lambda l: l.npoints > self.nb_pts, executor=MPIPoolExecutor())
        # else:
        runner = adaptive.BlockingRunner(learner, goal=lambda l: l.npoints > self.nb_pts)

        data = learner.data
        k_vec = np.fromiter(data.keys(), dtype=np.float)
        vals = np.array([data[k] for k in data.keys()])
        ind = np.argsort(k_vec)
        self.k_vec = k_vec[ind]
        vals = vals[ind]

        self.wf_val = vals

    def get_interpolators(self, interpolator=sc.interpolate.Akima1DInterpolator):
        self.wf_interp = [[[[None,None] for site in range(self.nb_sites)] for channel in range(self.nb_channels)] for lead in range(self.nb_leads)]
        for lead in range(self.nb_leads):
            for channel in range(self.nb_channels):  # caveat: at different energies a different number of channels can be opened!
                for site in range(self.nb_sites):
                    self.wf_interp[lead][channel][site][0] = interpolator(self.k_vec, self.wf_val[:, lead, channel, site, 0])
                    self.wf_interp[lead][channel][site][1] = interpolator(self.k_vec, self.wf_val[:, lead, channel, site, 1])

    def __call__(self, k, lead, channel, site):
        return self.wf_interp[lead][channel][site][0](k) + 1j*self.wf_interp[lead][channel][site][1](k)
