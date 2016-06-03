import numpy as np


try:
    import numba
    
    @numba.njit
    def _to_spherical(lat, lon, a, b, c):
        """Convert degrees latitude/longitude to coordinates on an ellipsoid."""
        x_y_z = np.zeros(lat.shape + (3,))
        for idx in np.ndindex(lat.shape):
            xyz = x_y_z[idx]
            la = lat[idx] * (np.pi / 180.)
            lo = lon[idx] * (np.pi / 180.)
            cos_la = np.cos(la)
            xyz[0] = a * cos_la * la
            xyz[1] = b * cos_la * np.sin(lo)
            xyz[2] = c * np.sin(la)

        return x_y_z
except:
    raise NotImplementedError("numba must be installed")


def to_spherical(lat, lon, ellipsoid=None):
    """Convert geographic coordinates to coordinates on an ellipsoid.
    
    Arguments:
        lat - numpy array
            Geographic latitude coordinates in range (-90, 90)
        
        lon - numpy array same size as 'lat'
            Geographic longitude coordinates in range (-180, 180)
        
        ellipsoid - [None] | 'WGS84' | 'SPHERE' | tuple(float)
            Specify a reference ellipsoid by name. Alternatively, the
            half-length of the axes along each dimension can provided
            in a tuple of floats.
    
    Note:
        x and y are coordinates along the equatorial plane.
        Results are only valid for -90<lat<90 and -180<lon<180.
    
    References:
        <https://en.wikipedia.org/wiki/Ellipsoid>
    """
    if ellipsoid is None:
        a = b = c = 1.0
    
    elif ellipsoid in {'SPHERE', 'sphere'}:
        a, b, c = 6371007.181000
    
    elif ellipsoid in {'WGS84', 'wgs84'}:
        a, c =  6378137.0, 6356752.314245
        b = a
    
    elif np.isscalar(ellipsoid) or len(ellipsoid) is 1:
        a = ellipsoid, b = ellipsoid, c = ellipsoid
    
    elif len(ellipsoid) is 2:
        a, b = ellipsoid
        c = b
    
    elif len(ellipsoid) is 3:
        a, b, c = ellipsoid
    
    return _to_spherical(lat, lon, a, b, c)


class GeoTree:
    def __init__(self, lat, lon, ellipsoid=None, compact_nodes=False, balanced_tree=False):
        """Build a KDTree using reference grid coordinates."""
        import scipy
        import scipy.spatial
        assert lat.shape == lon.shape
        
        lat, lon = np.array(lat), np.array(lon)
        self.s = lat.shape
        xyz = to_spherical(np.ravel(lat),np.ravel(lon), ellipsoid=ellipsoid)
        
        self.tree = scipy.spatial.cKDTree(xyz, leafsize=50, compact_nodes=compact_nodes, 
                                          balanced_tree=balanced_tree)
    
    def query(self, lat, lon, ellipsoid=None, ):
        """Find the closest ref coord to each of the query coords."""
        lat, lon = np.array(lat), np.array(lon)
        s = lat.shape
        lat, lon = np.ravel(lat), np.ravel(lon)
        xyz = to_spherical(lat, lon, ellipsoid=ellipsoid)
        distances, indices = self.tree.query(xyz)
        # The indices of the lat/lon grid that was inputted
        ret = np.unravel_index(indices, self.s)
        return distances, ret