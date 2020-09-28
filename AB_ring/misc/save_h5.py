import h5py
import numpy as np


def save_to_hdf(filename, param, t, gL, gG):
    """Saves g(t)< and g(t)> into a hdf5 compatible with `g0_model` in Keldy.

    param : same dictionary structure as in `g0_model` in Keldy
    gL : g(t)<, shape (N, 2, 2), complex
    gG : g(t)>, shape (N, 2, 2), complex

    TODO: shape of gL, gG could change (e.g. (N,3,3))
    TODO: spin up/down could be different."""

    def add_mesh(dct, t):
        mesh = dct.create_group("mesh")
        mesh["max"] = np.max(t)
        mesh["min"] = np.min(t)
        mesh["size"] = len(t)
        write_attr_string(mesh, "TRIQS_HDF5_data_scheme", "MeshReTime")

    def write_data_string(dct, key, s):
        dct.create_dataset(key, shape=(), data=np.array(s, dtype="S%d" % len(s)), dtype="|S%d" % (1+len(s)) )
    def write_attr_string(dct, key, s):
        dct.attrs.create(key, shape=(), data=np.array(s, dtype="S%d" % len(s)), dtype="|S%d" % (1+len(s)) )

    with h5py.File(filename, "a") as f:
        g0 = f.create_group("g0")

        # Add parameters and other things
        g0["make_dot_lead"] = param["make_dot_lead"]

        for err in ["greater_ft_error", "lesser_ft_error"]:
            g_err = g0.create_group(err)
            g_err["data"] = np.empty(shape=(0, 0, 0, 2), dtype=np.double)
            add_mesh(g_err, np.array([0.0,1.0], dtype=np.double))

        g0_param = g0.create_group("model_omega").create_group("param_")
        for (key, val) in param.items():
            if key == "bath_type" or key == "ft_method":
                continue
            g0_param[key] = val

        write_data_string(g0_param, "bath_type", param["bath_type"])
        write_data_string(g0_param, "ft_method", param["ft_method"])


        # Add g<,>
        g0_lesser = g0.create_group("g0_lesser")
        g0_greater = g0.create_group("g0_greater")
        ## Add common data
        for g in [g0_lesser, g0_greater]:
            for spin in ["up", "down"]:
                g.create_group(spin)
                add_mesh(g[spin], t)

                indices = g[spin].create_group("indices")
                for ind in ["left", "right"]:
                    indices[ind] = np.array([str(i) for i in range(gL.shape[-1])], dtype='|S2') # TODO this should match dimension of g<, g>

            g["block_names"] = np.array(["up", "down"], dtype="|S5")

        ## Add g<> data
        for spin in ["up", "down"]:
            assert gL.shape == gG.shape
            shape = list(gL.shape) + [2]
            
            dset = g0_lesser[spin].create_dataset("data", shape, compression="gzip", compression_opts=9, dtype="d")
            data = np.concatenate((gL[..., np.newaxis].real, gL[..., np.newaxis].imag), axis=-1 ).real
            dset[:] = data[:]
            
            dset = g0_greater[spin].create_dataset("data", shape, compression="gzip", compression_opts=9, dtype="d")
            data = np.concatenate((gG[..., np.newaxis].real, gG[..., np.newaxis].imag), axis=-1 ).real
            dset[:] = data[:]


        # Add attributes
        write_attr_string(g0, "TRIQS_HDF5_data_scheme", "KELDY_G0Model")
        for g in [g0_lesser, g0_greater]:
            write_attr_string(g, "TRIQS_HDF5_data_scheme", "BlockGf")
            for spin in ["up", "down"]:
                write_attr_string(g[spin], "TRIQS_HDF5_data_scheme", "Gf")
                write_attr_string(g[spin]["data"], "__complex__", "1")
                write_attr_string(g[spin]["indices"], "TRIQS_HDF5_data_scheme", "GfIndices")

        for err in ["greater_ft_error", "lesser_ft_error"]:
            write_attr_string(g0[err], "TRIQS_HDF5_data_scheme", "Gf")
            write_attr_string(g0[err]["data"], "__complex__", "1")
