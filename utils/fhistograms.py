"""
Wrapper for FHistograms between binary objects.
"""

import ctypes
import os
import numpy as np

# Load C shared library for FHistograms
libfh = os.path.join(os.path.dirname(__file__), 'libfhistograms_raster.so')
clib = ctypes.cdll.LoadLibrary(libfh)


def fhistogram(a, b=None, n_dirs=180, force_type=0.0):
    """
    Compute a Force Histogram between two binary images.

    The FHistogram is computed between `a` and `b` images, along `n_dirs`
    directions and using the attraction force `force_type`.

    Parameters
    ----------
    a, b : (w, h,) array_like
        The two binary images to consider. `a` and `b` must have the same size.
        If `b` is None, the FHistogram is computed between `a` and `a`.
    n_dirs : int
        The number of directions to consider. Default is 180.
    force_type : float
        The attraction force used to compute the FHistogram. Default is 0.0.

    Returns
    -------
    fh : (n_dirs,) ndarray
        The FHistogram between `a` and `b` along `n_dirs` directions using the
        attraction force `force_type`.

    Notes
    -----
    The FHistogram is computed using a C shared library called with ctypes.

    fhistogram(a, b) represents the spatial position of `a` relative to `b`.
    fhistogram(a, a) is the FHistogram representing the shape of `a`.
    fhistogram(a) is equivalent to fhistogram(a, a).

    The attraction force `force_type` must be < 1 when images are overlapping.

    """
    if b is None:
        b = a
    if a.shape != b.shape:
        raise ValueError('a and b must have the same shape.')
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError('a and b must be 2D with one channel.')
    if b is a and force_type >= 1:
        raise ValueError('0 <= force_type < 1 when b == a.')

    # # Adjacent objects can raise discretization issues when force_type != 0
    # if erode and force_type != 0:
    #     # Perform small erosions on objects while they overlap or are adjacent
    #     while (np.any(np.logical_and(dilation(a, selem=square(3)),
    #                                  dilation(b, selem=square(3))))):
    #         a = erosion(a, selem=square(3))
    #         b = erosion(b, selem=square(3))

    # Compute and return the FHistogram between a and b
    fhistogram = np.ndarray(n_dirs)
    height, width = a.shape
    clib.FRHistogram_CrispRaster(
        fhistogram.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n_dirs),
        ctypes.c_double(force_type),
        a.ctypes.data_as(ctypes.c_char_p),
        b.ctypes.data_as(ctypes.c_char_p),
        ctypes.c_int(width),
        ctypes.c_int(height))
    return fhistogram
