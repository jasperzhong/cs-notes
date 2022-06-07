import perfplot
import pickle
import numpy
import h5py
import tables
import zarr


def npy_write(data):
    numpy.save("npy.npy", data)


def hdf5_write(data):
    f = h5py.File("hdf5.h5", "w")
    f.create_dataset("data", data=data)


def pickle_write(data):
    with open("test.pkl", "wb") as f:
        pickle.dump(data, f)


def pytables_write(data):
    f = tables.open_file("pytables.h5", mode="w")
    gcolumns = f.create_group(f.root, "columns", "data")
    f.create_array(gcolumns, "data", data, "data")
    f.close()


def zarr_write(data):
    zarr.save("out.zarr", data)


perfplot.save(
    "write.png",
    setup=numpy.random.rand,
    kernels=[npy_write, hdf5_write, pickle_write, pytables_write, zarr_write],
    n_range=[2 ** k for k in range(28)],
    xlabel="len(data)",
    equality_check=None,
)
