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
        a = b = c = 6371007.181000 # meters
    
    elif ellipsoid in {'WGS84', 'wgs84'}:
        a, c =  6378137.0, 6356752.314245 # meters
        b = a
    
    elif np.isscalar(ellipsoid) or len(ellipsoid) is 1:
        a = b = c = ellipsoid
    
    elif len(ellipsoid) is 2:
        a, b = ellipsoid
        c = b
    
    elif len(ellipsoid) is 3:
        a, b, c = ellipsoid
    
    return _to_spherical(lat, lon, a, b, c)


class GeoTree:
    """A kd-tree for geographic coordinates. Requires numpy and scipy"""
    
    def __init__(self, lat, lon, ellipsoid=None, leafsize=32,
                 compact_nodes=True, balanced_tree=False):
        """Build a KDTree using reference grid coordinates.
        
        Arguments:
            lat - numpy.array(shape=S)
                A grid of latitude values that we want to sample *TO*.
                
            lat - numpy.array(shape=S)
                A grid of longitude values that we want to sample *TO*.
            
            ellipsoid - [None] | "WGS84" | "SPHERE" | sequence of axis half-lengths
                Specifies the shape of the Earth/sphere/geoid that the lat, lon
                values are mapped on.
            
            compact_nodes - [False] | True
                See documentation [1].
            
            balanced_tree - [False] | True
                See documentation [1].
        
        References:
            [1] - scipy.spatial.cKDTree documentation:
            <http://docs.scipy.org/doc/scipy-0.17.0/reference/
            generated/scipy.spatial.cKDTree.html>
        """
        import scipy
        import scipy.spatial
        
        # Convert to numpy array, but do not check that sizes match
        lat, lon = np.array(lat), np.array(lon)
        self.S, self.ellipsoid = lat.shape, ellipsoid
        
        # Convert lat and lon to flattened views
        xyz = to_spherical(np.ravel(lat), np.ravel(lon), ellipsoid=ellipsoid)
        
        self.tree = scipy.spatial.cKDTree(xyz, leafsize=leafsize,
                                          compact_nodes=compact_nodes, 
                                          balanced_tree=balanced_tree)
    
    def query(self, lat, lon, ellipsoid=None, **kwargs):
        """Find the closest ref coord to each of the query coords.
        
        Arguments:
            lat - numpy.array(shape=T)
                Query coordinates that we want to convert *FROM*.
            
            lon - numpy.array(shape=T)
                Query coordinates that we want to convert *FROM*.
            
            ellipsoid - [None] | "WGS84" | "SPHERE" | sequence of axis half-lengths
                Specifies the shape of the Earth/sphere/geoid that the lat, lon
                values are mapped on.
            
            distance_upper_bound - [None] | positive float
                If specified, only reference coordinates within this distance
                to other coordinates are returned.
        
        Returns:
            If `distance_upper_bound` is not specified then a 2-tuple with
            the following fields is returned:
            
            [0] numpy.array
                The first element is an array of shape T containing
                the distances from each query coordinate to the closest
                reference coordinate.
            
            [1] tuple(numpy.array(size=T, dtype=int))
                The indices of the reference coordinates that each query 
                coordinate corresponds to. The length this element is equal
                to the number of dimensions 
            
            If `distance_upper_bound` is specified then a 3-tuple is returned.
            In this case some of the query points may be invalid, in which case
            the closest ref point is undefined. We must exclude those by 
            providing an index to only the query points that have valid
            corresponding ref point.
            
            [0] tuple(numpy.array(size=(V,)))
                The indices of the query lat/lon coordinates that are valid
                points. Here the integer V is the number of "valid" points
                (i.e. query points that have a corresponding ref point within
                the specified `distance_upper_bound`. The length of this
                element is the dimensionality of the query points.
            
            [1] numpy.array(shape=(V,), dtype=float)
                The distance from each valid query point to the closest ref
                point.
            
            [2] tuple(numpy.array(shape=(V,), dtype=int))
                The distance from each valid query point to the closest ref
                point.
                
            
        """
        ellipsoid = ellipsoid or self.ellipsoid
        lat, lon = np.array(lat), np.array(lon)
        assert lat.shape == lon.shape
        T = lat.shape
        lat, lon = np.ravel(lat), np.ravel(lon)
        xyz = to_spherical(lat, lon, ellipsoid=ellipsoid)
        distances, indices = self.tree.query(xyz, **kwargs)
        
        if "distance_upper_bound" in kwargs:
            # Select only the valid indices to unravel
            valid = np.where(indices < self.tree.n)[0]
            indices = indices[valid]
            distances = distances[valid]
            return (np.unravel_index(valid, T),
                    distances,
                    np.unravel_index(indices, self.S))
        
        # Convert to indices of the lat/lon grid it was constructed with
        indices = np.unravel_index(indices, self.S)
        return distances.reshape(T), tuple(idx.reshape(T) for idx in indices)

