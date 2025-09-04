# front_tracking.py
"""
Front threshold-crossing detection and trajectory stitching.

Public API:
- Trajectory
- find_threshold_crossings
- find_closest_traj
- connect_crossing_points
- get_front_trajectories
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np

__all__ = [
    "Trajectory",
    "find_threshold_crossings",
    "find_closest_traj",
    "connect_crossing_points",
    "get_front_trajectories",
]


class Trajectory:
    """
    Class representing a single trajectory of a front, defined by its
    positions (x), corresponding row indices (idx), and times (t).
    New points can be appended as the trajectory evolves in time.

    Attributes
    ----------
    x : np.ndarray of float
        Array of positions (values) along the trajectory.
    idx : np.ndarray of int
        Array of row indices corresponding to each position in ``x``.
    t : np.ndarray of float
        Array of times corresponding to each row index (``t = idx * dt``).
    dt : float
        Time step used to convert indices to times.
    """
    def __init__(self,first_value, first_idx, dt):
        self.x   = np.array([first_value], dtype = np.float64);
        self.idx = np.array([first_idx], dtype = np.int64);
        self.dt  = float(dt)
        self.t   = self.idx * self.dt
        
    def add_point(self, value, idx):
        """Append a new point (value, idx) to the trajectory."""
        self.x   = np.append(self.x,   value)
        self.idx = np.append(self.idx, idx)
        self.t   = self.idx * self.dt
        
    def last_idx(self):
        return self.idx[-1]
    
    def last_value(self):
        return self.x[-1]     

def find_threshold_crossings(A, x, a):
    """
    Find all x where A[n, :] crosses the level 'a' for each row n.
    Uses linear interpolation between adjacent grid points.

    Parameters
    ----------
    A : (N, M) array
        Values of the quantity at N samples and M gridpoints.
    x : (M,) array
        Monotonic grid (not necessarily uniform).
    a : float
        Threshold level.

    Returns
    -------
    x_cross : (K,) float array
       x-locations of all crossings over all rows.
   rows : (K,) int array
        Index of the row of the crossing. Such that row[i] is the row at
        which x_cross[i] has been found.
    """
    
    # Adjacent values
    L = A[:, :-1]
    R = A[:, 1:]
    denom = R - L
    
    # Crossing mask: strict sign change around 'a'
    BL = L - a
    BR = R - a
    cross_mask = (BL * BR) < 0 
    cross_mask &= (denom != 0) 
    
    # Indices of crossing edges
    rows, cols = np.where(cross_mask)
    
    #In the above, rows will be the index of the time sample and 
    # cols is the index to the LEFT of the crossing. To obtain the 
    # exact x location we interpolate
    
    #Linear interpolation
    frac    = (a - L[rows, cols])/denom[rows, cols]
    x_cross = x[cols] + frac*(x[cols+1] - x[cols])
    
    return x_cross, rows

def find_closest_traj(traj_list, point, threshold):
    """
    Find the trajectory in traj_list whose last_value() is closest to 'point',
    provided the distance is below a threshold.

    Parameters
    ----------
    traj_list : list of Trajectory
        List of trajectory objects, each with a .last_value() method.
    point : float
        Target value to compare against.
    threshold : float
        Maximum allowed distance between trajectory.last_value() and 'point'.

    Returns
    -------
    Trajectory or None
        The trajectory with minimal distance to 'point' within 'threshold',
        or None if no trajectory satisfies the condition.
    """
    
    # compute distances
    diffs = np.array([abs(A.last_value() - point) for A in traj_list])
    # mask by threshold
    mask = diffs <= threshold
    if not np.any(mask):
        return None  # nothing within threshold
    # index of minimal difference among those
    i_best = np.argmin(np.where(mask, diffs, np.inf))
    return traj_list[i_best]

def connect_crossing_points(x_cross, rows, x, d_max,dt):
    """
    Build trajectories of crossing points across rows by connecting
    nearest neighbors between consecutive rows, with a maximum allowed jump.

    Parameters
    ----------
    x_cross : (K,) array_like
        Crossing positions (floats).
    rows : (K,) array_like
        Row indices corresponding to each crossing in x_cross.
    x : (M,) array_like
        Grid coordinates (not directly used here, but can be useful for context).
    d_max : float
        Maximum allowed |Δx| between consecutive rows to connect a crossing
        to an existing trajectory. If no trajectory is close enough, a new one is started.

    Returns
    -------
    traj_list : list of Trajectory
        List of Trajectory objects. Each trajectory contains arrays of
        positions (x) and row indices (idx), built by connecting crossings
        across rows.
    """
    
    unique_rows = np.unique(rows); 
    traj_list    = [];   #All trajectories
    for r in unique_rows:
        #find all of the crossing values at the current row
        cross_values_at_level = x_cross[rows == r]; 
        for c in cross_values_at_level:
            #find all the active trajectores
            active_traj = [A for A in traj_list if A.last_idx() == (r-1)]
            closest_match = find_closest_traj(active_traj, c, d_max);
            if closest_match is None: 
                new_traj = Trajectory(c,r,dt); #create new trajectory
                traj_list.append(new_traj);
            else:
                closest_match.add_point(c,r);
    return traj_list
            

def get_front_trajectories(A, x, a, d_max,dt):
    """
    Extract front trajectories from a 2D field A by finding threshold crossings
    and connecting them across rows into trajectories.

    Parameters
    ----------
    A : (N, M) array_like
        Values of the quantity at N samples and M gridpoints.
        Each row corresponds to a different time/sample index.
    x : (M,) array_like
        Grid coordinates (monotonic, not necessarily uniform).
    a : float
        Threshold value. Crossings are detected where A crosses this level.
    d_max : float
        Maximum allowed |Δx| between consecutive rows to connect crossings
        into a single trajectory.
    dt : float
            Time step used to convert indices to times.

    Returns
    -------
    traj_list : list of Trajectory
        List of Trajectory objects, each storing the positions (x) and
        indices (row numbers) and times (t) of a connected front trajectory.
    """
    x_cross, rows = find_threshold_crossings(A, x, a);
    traj_list = connect_crossing_points(x_cross, rows, x, d_max,dt)
    return traj_list
