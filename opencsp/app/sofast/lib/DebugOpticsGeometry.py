"""Class holding debug information for proces_optics_geometry calculations"""


class DebugOpticsGeometry:
    """Class for holding debug data for used in 'process_optics_geometry"""

    def __init__(self):
        self.debug_active: bool = False
        self.figures: list = []
