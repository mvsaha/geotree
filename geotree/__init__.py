import numpy as np


try:
    import numba
    
    @numba.njit
    def _to_spherical(lat, lon, a, b, c):
        """Convert degrees latitude/longitude to coordinates on an ellipsoid.
        
        About ~25% Faster than the numpy version (useful for big arrays).
        """
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

except ImportError:
    
    def _to_spherical(lat, lon, a, b, c):
        la = (np.pi / 180) * lat
        lo = (np.pi / 180) * lon
        cos_la = np.cos(la)
        x = a * cos_la * la
        y = b * cos_la * np.sin(lo)
        z = c * np.sin(la)
        return np.concatenate((x[...,None],
                               y[...,None],
                               z[...,None]), axis=-1).shape


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
            in a tuple of floats. The default value for ellipsoid is 
            None, in which case it defaults to the unit sphere.
    
    Note:
        x and y are coordinates along the equatorial plane.
        Results are only valid for -90<lat<90 and -180<lon<180.
    
    References:
        <https://en.wikipedia.org/wiki/Ellipsoid>
    """
    if ellipsoid is None:
        # Put points onto a unit sphere
        a = b = c = 1.0
    
    elif ellipsoid in {'SPHERE', 'sphere'}:
        a, b, c = 6371007.181000 # meters
    
    elif ellipsoid in {'WGS84', 'wgs84'}:
        a, c =  6378137.0, 6356752.314245 # meters
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
    """A kd-tree for geographic coordinates."""
    
    def __init__(self, lat, lon, ellipsoid=None, compact_nodes=False, balanced_tree=False):
        """Build a KDTree using reference grid coordinates.
        
        Arguments:
            lat - numpy.array(shape=(m,n,o,...))
                A grid of latitude values that we want to sample *TO*.
                
            lat - numpy.array(shape=(m,n,o,...))
                A grid of longitude values that we want to sample *TO*.
            
            ellipsoid - [None] | "WGS84" | "SPHERE" | sequence of axis half-lengths
                Specifies the shape of the Earth/sphere/geoid that the lat, lon
                values refer to.
            
            
        """
        import scipy
        import scipy.spatial
        assert lat.shape == lon.shape
        
        lat, lon = np.array(lat), np.array(lon)
        self.s = lat.shape
        xyz = to_spherical(np.ravel(lat),np.ravel(lon), ellipsoid=ellipsoid)
        
        self.tree = scipy.spatial.cKDTree(xyz, leafsize=50, compact_nodes=compact_nodes, 
                                          balanced_tree=balanced_tree)
    
    def query(self, lat, lon, **kwargs):
        """Find the closest ref coord to each of the query coords.
        
        
        
        
        """
        if "ellipsoid" not in kwargs:
            ellipsoid = None
        
        lat, lon = np.array(lat), np.array(lon)
        s = lat.shape
        lat, lon = np.ravel(lat), np.ravel(lon)
        xyz = to_spherical(lat, lon, ellipsoid=ellipsoid)
        distances, indices = self.tree.query(xyz, )
        # The indices of the lat/lon grid that was inputted
        ret = np.unravel_index(indices, self.s)
        return distances, ret