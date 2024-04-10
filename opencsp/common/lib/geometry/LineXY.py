import numpy as np
from numpy.random import RandomState, SeedSequence, MT19937
from scipy.optimize import minimize

from opencsp.common.lib.geometry.Vxy import Vxy


class LineXY:
    def __init__(self, A: float, B: float, C: float):
        """
        Representation of a homogenous line with the following properties:
            - General form: Ax + By + C = 0
            - Slope intercept form: y = -A/Bx - C/B
            - XY vector (A, B) (Line.n_vec) is normal unit vector pointing
              perpendicular to the line. The AB vector is normalized upon
              instantiation.
            - C is the distance along the normal vector from the line to the
              origin. C can be positive or negative.

        Parameters
        ----------
        A,B,C : float
            Line coefficients.

        Returns
        -------
        LineXY.

        """
        # Normalize AB vector
        AB = np.array([A, B])
        mag = np.linalg.norm(AB)

        # Save in class
        self.A = A / mag
        self.B = B / mag
        self.C = C / mag

    def __repr__(self):
        return '2D Line: ' + self.A.__repr__() + ', ' + self.B.__repr__() + ', ' + self.C.__repr__()

    @property
    def n_vec(self) -> Vxy:
        """
        Returns normal vector to line.

        Returns
        -------
        Vxy
            Normal vector to line.

        """
        return Vxy([self.A, self.B])

    @property
    def ABC(self) -> np.ndarray:
        """
        Returns ABC coefficients in length 3 ndarray.

        Returns
        -------
        ndarray
            ABC coefficiens in array.

        """
        return np.array([self.A, self.B, self.C])

    @property
    def slope(self) -> float:
        """Get the slope of the line (could be infinity!)"""
        if abs(self.B) < 1e-10:
            return np.inf
        return -self.A / self.B

    @classmethod
    def fit_from_points(cls, Pxy: Vxy, seed: int = 1, neighbor_dist: float = 1.0):
        """
        Fits a LineXY to a set of points using Ransac method.

        Parameters
        ----------
        Pxy : Vxy
            XY points to fit line to.
        seed : int, optional
            Random number generator seed. The default is 1.
        neighbor_dist : float, optional
            Threshold distance to use in Ransac algorithm. The default is 1..

        Returns
        -------
        LineXY

        """
        # Reset random number generator
        rs = RandomState(MT19937(SeedSequence(seed)))

        # If greater than N points are included, stop searching
        n = len(Pxy)
        thresh = int(0.99 * n)
        if n <= 15:
            raise ValueError(f'To fit line from points, must have > 15 points, but {n:d} were given.')

        # Fit from random combinations of points, keeping the best
        best_active = 0
        for idx in range(1000):
            # Find two separate points
            i1 = np.floor(rs.rand() * n).astype(int)
            i2 = np.floor(rs.rand() * n).astype(int)
            while Pxy.data[0, i1] == Pxy.data[0, i2] and Pxy.data[1, i1] == Pxy.data[1, i2]:
                i2 = np.floor(rs.rand() * n).astype(int)

            # Fit line to two poins
            p1 = Pxy[i1]
            p2 = Pxy[i2]
            Lxy = LineXY.from_two_points(p1, p2)

            # Calculate point distances away from line
            ds = Lxy.dist_from(Pxy)
            mask_neighbor = ds < neighbor_dist
            n_active = mask_neighbor.sum()

            # Keep line with most point matches
            if n_active > best_active:
                # Save active point mask for best fit line
                active_mask = np.copy(mask_neighbor)
                best_active = np.copy(n_active)

                # Save starting point for best fit line
                A_active = np.copy(Lxy.A)
                B_active = np.copy(Lxy.B)
                C_active = np.copy(Lxy.C)

            # Check if more than N points were used
            if n_active > thresh:
                break

        # Define mean squares error function
        def error_func(X: tuple, Pxy: Vxy):
            # Unpack inputs
            theta, C = X
            A = np.sin(theta)
            B = np.cos(theta)

            # Instantiate line
            L = LineXY(A, B, C)

            # Calculate RMS error
            dists = L.dist_from(Pxy)
            return np.mean(dists**2)

        # Least squares linear fit to active points
        Pxy_active = Pxy[active_mask]
        theta = np.arctan2(A_active, B_active)
        out = minimize(error_func, np.array([theta, C_active]), args=(Pxy_active,))

        # Create fitted line
        A = np.sin(out.x[0])
        B = np.cos(out.x[0])
        C = out.x[1]
        return LineXY(A, B, C)

    @classmethod
    def from_two_points(cls, Pxy1: Vxy, Pxy2: Vxy):
        """
        Creates XY line from 2 points.

        Parameters
        ----------
        Pxy1 : Vxy
            First point.
        Pxy2 : Vxy
            Second point.

        Returns
        -------
        LineXY

        """
        if len(Pxy1) != 1 or len(Pxy2) != 1:
            raise ValueError('Input vectors must be length 1.')

        # Find line coefficients
        delta = Pxy2 - Pxy1
        R = np.array([[0, 1], [-1, 0]])  # CW 90 deg

        AB = np.matmul(R, delta.data).squeeze()
        C = -np.dot(Pxy1.data.squeeze(), AB)

        ABC = np.concatenate((AB, [C]), axis=0)

        return LineXY(*ABC)

    def y_from_x(self, xs: np.ndarray | float) -> np.ndarray | float:
        """
        Returns y points that lie on line given corresponding x points.

        Parameters
        ----------
        xs : ndarray
            X points.

        Returns
        -------
        ndarray
            Y points.

        """
        return (-self.A * xs - self.C) / self.B

    def x_from_y(self, ys: np.ndarray | float) -> np.ndarray | float:
        """
        Returns x points that lie on line given corresponding y points.

        Parameters
        ----------
        ys : ndarray
            Y points.

        Returns
        -------
        ndarray
            X points.

        """
        return (-self.B * ys - self.C) / self.A

    def dist_from(self, Pxy: Vxy) -> np.ndarray:
        """
        Calculates perpendicular distance magnitude from line to XY points.

        Parameters
        ----------
        Pxy : Vxy
            Input points.

        Returns
        -------
        ndarray
            1d ndarray with length: len(Pxy).

        """
        return np.abs(self.dist_from_signed(Pxy))

    def dist_from_signed(self, Pxy: Vxy) -> np.ndarray:
        """
        Calculates perpendicular distance from line to XY points. Distances are
        positive if in same direction from line as vector [A, B].

        Parameters
        ----------
        Pxy : Vxy
            Input points.

        Returns
        -------
        ndarray
            1d ndarray with length: len(Pxy).

        """
        return Pxy.dot(self.n_vec) + self.C

    def intersect_with(self, Lxy: 'LineXY') -> Vxy | None:
        """
        Calculates intersection point of two 2D lines.

        Parameters
        ----------
        Lxy : LineXY
            2D line to intersect with current line.

        Returns
        -------
        Vxy
            2D intersection point. None if Lxy is parallel to this line.

        """
        # test if the two lines are parallel
        # horizontal line A=0, vertical line B=0
        if (
            (abs(self.A) < 1e-10 and abs(Lxy.A) < 1e-10)
            or (abs(self.B) < 1e-10 and abs(Lxy.B) < 1e-10)
            or (abs(self.slope - Lxy.slope) < 1e-10)
        ):
            return None

        # find the intersection
        v = np.cross(self.ABC, Lxy.ABC)
        return Vxy(v[:2] / v[2])

    def flip(self):
        """
        Flips the orientation of a line. Returns a flipped copy of the LineXY.

        Returns
        -------
        LineXY
            Flipped line.

        """
        L_out = LineXY(*self.ABC)
        L_out.flip_in_place()
        return L_out

    def flip_in_place(self) -> None:
        """
        Flips the orientation of a line. The objects's coefficients are
        updated internally.

        """
        self.A *= -1
        self.B *= -1
        self.C *= -1
