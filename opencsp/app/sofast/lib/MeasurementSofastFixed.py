import datetime as dt

import numpy as np

import opencsp.app.sofast.lib.AbstractMeasurementSofast as ams
import opencsp.app.sofast.lib.DistanceOpticScreen as osd
from opencsp.common.lib.geometry.Vxy import Vxy
import opencsp.common.lib.tool.hdf5_tools as hdf5_tools


class MeasurementSofastFixed(ams.AbstractMeasurementSofast):
    """Fixed pattern measuremnt class. Stores measurement data. Can save
    and load to HDF file format.
    """

    def __init__(
        self,
        image: np.ndarray,
        dist_optic_screen_measure: osd.DistanceOpticScreen,
        origin: Vxy,
        date: dt.datetime = None,
        name: str = '',
    ):
        """Saves measurement data in class.

        Parameters
        ----------
        image : np.ndarray
            (M, N) ndarray, measurement image
        dist_optic_screen_measure : DistanceOpticScreen
            Measurement point on the optic, and distance from that point to the screen
        origin : Vxy
            The centroid of the origin dot, pixels
        date : datetime, optional
            Collection date/time. Default is dt.datetime.now()
        name : str, optional
            Name or serial number of measurement. Default is empty string ''
        """
        super().__init__(dist_optic_screen_measure, date, name)
        self.image = image
        self.origin = origin

    @classmethod
    def load_from_hdf(cls, file: str, prefix='') -> 'MeasurementSofastFixed':
        """
        Loads from HDF file

        Parameters
        ----------
        file : string
            HDF file to load

        """
        # Load grid data
        datasets = [prefix + 'MeasurementSofastFixed/image', prefix + 'MeasurementSofastFixed/origin']
        kwargs = hdf5_tools.load_hdf5_datasets(datasets, file)
        kwargs.update(super()._load_from_hdf(file, prefix + 'MeasurementSofastFixed'))

        kwargs['origin'] = Vxy(kwargs['origin'])

        return cls(**kwargs)

    def save_to_hdf(self, file: str, prefix='') -> None:
        """
        Saves to HDF file

        Parameters
        ----------
        file : string
            HDF file to save
        """
        datasets = [prefix + 'MeasurementSofastFixed/image', prefix + 'MeasurementSofastFixed/origin']
        data = [self.image, self.origin.data.squeeze()]

        # Save data
        hdf5_tools.save_hdf5_datasets(data, datasets, file)
        super()._save_to_hdf(file, prefix + 'MeasurementSofastFixed')
