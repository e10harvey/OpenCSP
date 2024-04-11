"""Parametric rectangular mirror wtih origin in center of rectangular region
representing a single reflective surface defined by an algebraic function.
"""

from typing import Callable

from opencsp.common.lib.geometry.RegionXY import RegionXY
from opencsp.common.lib.csp.MirrorParametric import MirrorParametric


class MirrorParametricRectangular(MirrorParametric):
    """
    Mirror implementation defined by a parametric function and rectangular side lengths.
    """

    def __init__(self, surface_function: Callable[[float, float], float], size: tuple[float, float] | float) -> None:
        """Instantiates a MirrorParametricRectangular object.

        Parameters
        ----------
        surface_function : Callable[[float, float], float]
            See MirrorParametric for details.
        size : tuple[float, float] | float
            The size of the mirror. If input type is 'float,' outputs square mirror with
            side lengths equal to size. If input length is 2, size is interpreted as
            size=(x, y) where x is the x side lengths and y is the y side lengths.
        """
        self.surface_function = surface_function

        # Use the XY size to make a rectangular region
        region = RegionXY.rectangle(size)

        # Instantiate mirror class
        super().__init__(surface_function, region)

        # Save width and height
        left, right, bottom, top = self.region.axis_aligned_bounding_box()
        self.width = right - left
        self.height = top - bottom
