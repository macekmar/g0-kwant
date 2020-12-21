import scipy as sc
import numpy as np
import kwant
import adaptive
from concurrent.futures import ProcessPoolExecutor


class WaveFunInterpolation():
    """Build an interpolation of a wave function using adaptive package.

    Adaptive: https://github.com/python-adaptive/adaptive
    Adaptive puts more points for the interpolation where changes are larger.
    At the moment, adaptive does not parallelize well on a cluster.

    Parameters
    ----------
    syst :
        Kwant FiniteSystem
    nb_pts :
        number of points for the interpolation
    eps :
        how close to the band edges we go, we sample k ∈ [ε, π - ε].
    gamma_wire :
        γ in the wires, needed for the dispertion relation ε(k).
    """
    def __init__(self, syst, nb_pts, eps, eps_i=0, gamma_wire=1.0):
        self.syst = syst
        self.nb_pts = nb_pts
        self.eps = eps
        self.eps_i = eps_i
        self.gamma_wire = gamma_wire
        self.nb_sites = self.syst.hamiltonian_submatrix().shape[0]
        self.nb_leads = len(self.syst.leads)
        self.nb_channels = 1
        self.k_vec = None
        self.wf_val = None
        self.wf_interp = None

    def wf(self, k):
        """ Calculates wave function at wave vector k.

        kwant calculates at some energy, whereas we provide wave vector k.
        Returns an array of size:
            nb_leads × nb_channels × nb_sites × 2 (Re, Im).
        """
        e = self.eps_i + 2*np.abs(self.gamma_wire)*np.cos(k)
        wf = kwant.wave_function(self.syst, energy=e)
        res = np.array([[wf(i).real, wf(i).imag] for i in range(self.nb_leads)])
        return np.moveaxis(res, 1, -1)  # Re, Im should be last axis

    def get_data(self, world_size=1):
        """Gathers data using adaptive package.

        Note: adaptive can use mpi but I have not managed to use it
        successfully. """
        learner = adaptive.Learner1D(self.wf, bounds=[self.eps, np.pi - self.eps])
        executor = ProcessPoolExecutor(max_workers=world_size)
        runner = adaptive.BlockingRunner(learner, executor=executor, goal=lambda l: l.npoints > self.nb_pts)
        # extract data
        data = learner.data
        k_vec = np.fromiter(data.keys(), dtype=np.float)
        vals = np.array([data[k] for k in data.keys()])
        ind = np.argsort(k_vec)
        self.k_vec = k_vec[ind]
        self.wf_val = vals[ind]

    def get_interpolators(self, interpolator=sc.interpolate.Akima1DInterpolator):
        """Builds interpolators from gathered data."""
        self.wf_interp = [[[
            [None, None] for site in range(self.nb_sites)]
            for channel in range(self.nb_channels)]
            for lead in range(self.nb_leads)]

        for lead in range(self.nb_leads):
            for channel in range(self.nb_channels):
                for site in range(self.nb_sites):
                    self.wf_interp[lead][channel][site][0] = interpolator(self.k_vec, self.wf_val[:, lead, channel, site, 0])
                    self.wf_interp[lead][channel][site][1] = interpolator(self.k_vec, self.wf_val[:, lead, channel, site, 1])

    def __call__(self, k, lead, channel, site):
        return self.wf_interp[lead][channel][site][0](k) + 1j*self.wf_interp[lead][channel][site][1](k)
