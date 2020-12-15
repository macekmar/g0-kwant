from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
import scipy as sc
import numpy as np
import kwant
import adaptive
from concurrent.futures import ProcessPoolExecutor



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
        
    def get_data(self, syst):
        # world = MPI.COMM_WORLD
        self.wf_val = np.zeros((len(self.engs), self.nb_leads, self.nb_channels, self.nb_sites), dtype=np.complex)
        for i, e in enumerate(self.engs):
            wf = kwant.wave_function(syst, energy=e)
            self.wf_val[i, ...] = np.array([wf(j) for j in range(self.nb_leads)])
        # self.wv_val = world.allreduce(wf_val)
        # print(self.wv_val)

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
    """Build an interpolation of a wave function using uniformly spaced points.
    
    This one is intended for the cluster.
    There is a discontinuity at k = π/2 (TODO: Is this always true, it is not the only one...)
    
    Parameters:
        syst : finalized Kwant system
        nb_pts : number of points for the interpolation
        eps : how close to the band edges we go, we sample in [-D + ε, D - ε].
        gamma_wire : γ in the wires, needed for the dispertion relation ε(k).
    """
    def __init__(self, syst, nb_pts, eps, world=MPI.COMM_WORLD, gamma_wire=1.0):
        self.syst = syst
        self.nb_pts = nb_pts
        self.eps = eps
        self.gamma_wire = gamma_wire
        self.nb_sites = self.syst.hamiltonian_submatrix().shape[0]
        self.nb_leads = 2
        self.nb_channels = 1
        self.k_lim = []
        self.pieces = []

        assert world.size % 2 == 0
        nb_left = world.size//2
        nb_right = world.size - nb_left
        intervals = np.concatenate((np.linspace(0, np.pi/2, nb_left+1),
                                    np.linspace(np.pi/2, np.pi, nb_right+1)))
        k_int = np.array([[i, self.eps] for i in intervals])

        k_int = np.delete(k_int, nb_left+1, axis=0)
        # if world.rank == 0:
        #     print(k_int)

        for i in range(1,len(k_int)):
            k_lim = [k_int[i-1][0] + k_int[i-1][1], k_int[i][0] - k_int[i][1]]
            k_vec = np.linspace(k_lim[0], k_lim[1], self.nb_pts)
            engs = 2*np.abs(self.gamma_wire)*np.cos(k_vec)
            self.k_lim.append(k_lim)
            self.pieces.append(fun_piece(k_vec, engs, self.nb_sites, self.nb_leads, self.nb_channels))
        for (k, eps_k) in k_int[1:-1,:]:
            # Linear interpolation between the cuts
            k_lim = [k - eps_k, k + eps_k]
            k_vec = np.linspace(k_lim[0], k_lim[1], 4)
            engs = 2*np.abs(self.gamma_wire)*np.cos(k_vec)
            self.k_lim.append(k_lim)
            self.pieces.append(fun_piece(k_vec, engs, self.nb_sites, self.nb_leads, self.nb_channels, interpolator=sc.interpolate.interp1d))
        
    def get_data(self, world):
        for i in range(len(self.pieces)):
            if i % world.size == world.rank: 
                self.pieces[i].get_data(self.syst)
        world.barrier()
        for i in range(len(self.pieces)):
            self.pieces[i] = world.bcast(self.pieces[i], root=i % world.size )
    
    def get_interpolators(self, world):
        for i in range(len(self.pieces)):
            # if world.rank == 0:
            #     print(i)
            #     print(self.pieces[i].k)
            self.pieces[i].get_interpolators()


    def __call__(self, k, lead, channel, site):
        if np.isscalar(k):
            i = np.argmax([(k > k_[0]) and (k < k_[-1]) for k_ in self.k_lim])
            return self.pieces[i](k, lead, channel, site)
        else:
            test = np.vstack([np.logical_and((k > k_[0]),  (k < k_[-1])) for k_ in self.k_lim]).T
            indices = np.argmax(test, axis=1)
            res = np.zeros((len(k)),dtype=np.complex)
            for i in range(len(k)) :
                res[i] = self.pieces[indices[i]](k[i], lead, channel, site)
            return res


class wave_fun_adapt():
    """Build an interpolation of a wave function using adaptive package.
    
    Adaptive: https://github.com/python-adaptive/adaptive
    Adaptive puts more points for the interpolation where changes are larger.
    At the moment, adaptive does not parallelize well on a cluster.
    
    Parameters:
        syst : finalized Kwant system
        nb_pts : number of points for the interpolation
        eps : how close to the band edges we go, we sample k ∈ [ε, π - ε].
        gamma_wire : γ in the wires, needed for the dispertion relation ε(k).
    """
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

    def wf(self, k):
        """ Calculates wave function at wave vector k.

        kwant calculates at some energy, whereas we provide wave vector k.
        Returns an array of size: 
            nb_leads × nb_channels × nb_sites × 2 (Re, Im)."""

        e = 2*np.abs(self.gamma_wire)*np.cos(k)
        wf = kwant.wave_function(self.syst, energy=e)
        res = np.array([[wf(i).real, wf(i).imag] for i in range(self.nb_leads)])
        return np.moveaxis(res, 1, -1) #re, im should be last axis

    def get_data(self, mpi=False, max_workers=4):
        """Gathers data using adapative package.

        Note: adaptive can use mpi but I have not managed to use it
        successfully. """
        learner = adaptive.Learner1D(self.wf, bounds=[self.eps, np.pi - self.eps])
        if mpi:
            runner = adaptive.Runner(learner, goal=lambda l: l.npoints > self.nb_pts, shutdown_executor=True, executor=ProcessPoolExecutor(max_workers=max_workers))
            runner.ioloop.run_until_complete(runner.task)
        else:
            runner = adaptive.BlockingRunner(learner, goal=lambda l: l.npoints > self.nb_pts)

        data = learner.data
        k_vec = np.fromiter(data.keys(), dtype=np.float)
        vals = np.array([data[k] for k in data.keys()])
        ind = np.argsort(k_vec)
        self.k_vec = k_vec[ind]
        vals = vals[ind]

        self.wf_val = vals

    def get_interpolators(self, interpolator=sc.interpolate.Akima1DInterpolator):
        """Builds interpolators from gathered data."""
        self.wf_interp = [[[[None,None] for site in range(self.nb_sites)] for channel in range(self.nb_channels)] for lead in range(self.nb_leads)]
        for lead in range(self.nb_leads):
            for channel in range(self.nb_channels):  # caveat: at different energies a different number of channels can be opened!
                for site in range(self.nb_sites):
                    self.wf_interp[lead][channel][site][0] = interpolator(self.k_vec, self.wf_val[:, lead, channel, site, 0])
                    self.wf_interp[lead][channel][site][1] = interpolator(self.k_vec, self.wf_val[:, lead, channel, site, 1])

    def __call__(self, k, lead, channel, site):
        return self.wf_interp[lead][channel][site][0](k) + 1j*self.wf_interp[lead][channel][site][1](k)
