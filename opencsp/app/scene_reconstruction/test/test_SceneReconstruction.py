"""Tests the photogrammetric reconstruction of xyz marker positions.

Run this file to drive the test manually.
To create new test data, uncomment/comment the lines below.

"""

import os
from os.path import join
import unittest

import numpy as np

from opencsp.app.scene_reconstruction.lib.SceneReconstruction import SceneReconstruction
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.geometry.Vxyz import Vxyz


class TestSceneReconstruction(unittest.TestCase):
    @classmethod
    def setUpClass(cls, dir_input: str = None, dir_output: str = None):
        """Tests the SceneReconstruction process. If directories are None,
        uses default test data directory. All input files listed below must be
        on the dir_input path.

        Input Files  (in dir_input):
        ---------------------------
            - camera.h5
            - known_point_locations.csv
            - aruco_marker_images/*.JPG
            - point_pair_distances.csv
            - alignment_points.csv

        Expected Files (in dir_output):
        ------------------------------
            - point_locations.csv

        Parameters
        ----------
        dir_input : str
            Input/measurement file directory, by default, None
        dir_output : str
            Expected output file directory, by default None

        """
        if (dir_input is None) or (dir_output is None):
            # Define default data directories
            base_dir = os.path.dirname(__file__)
            dir_input = join(base_dir, 'data', 'data_measurement')
            dir_output = join(base_dir, 'data', 'data_expected')

        verbose = 0

        # Load components
        camera = Camera.load_from_hdf(join(dir_input, 'camera.h5'))
        known_point_locations = np.loadtxt(
            join(dir_input, 'known_point_locations.csv'), delimiter=',', skiprows=1
        )
        image_filter_path = join(dir_input, 'aruco_marker_images', '*.JPG')
        point_pair_distances = np.loadtxt(
            join(dir_input, 'point_pair_distances.csv'), delimiter=',', skiprows=1
        )
        alignment_points = np.loadtxt(
            join(dir_input, 'alignment_points.csv'), delimiter=',', skiprows=1
        )

        # Perform marker position calibration
        scene_recon = SceneReconstruction(
            camera, known_point_locations, image_filter_path
        )
        scene_recon.run_calibration(verbose)

        # Scale points
        point_pairs = point_pair_distances[:, :2].astype(int)
        distances = point_pair_distances[:, 2]
        scene_recon.scale_points(point_pairs, distances, verbose=verbose)

        # Align points
        marker_ids = alignment_points[:, 0].astype(int)
        alignment_values = Vxyz(alignment_points[:, 1:4].T)
        scene_recon.align_points(marker_ids, alignment_values, verbose=verbose)

        # Test results
        cls.pts_meas = scene_recon.get_data()
        cls.scene_recon = scene_recon
        cls.dir_output = dir_output

    def test_calibrated_corner_locations(self):
        """Tests relative corner locations"""
        pts_exp = np.loadtxt(
            join(self.dir_output, 'point_locations.csv'), delimiter=',', skiprows=1
        )
        np.testing.assert_allclose(self.pts_meas, pts_exp, atol=1e-5, rtol=0)
        print('Corner locations tested successfully.')

    def save_csv(self):
        """Saves CSV file of points to data location"""
        base_dir = os.path.dirname(__file__)
        file = join(base_dir, 'data/data_expected/point_locations.csv')
        self.scene_recon.save_data_as_csv(file)


if __name__ == '__main__':
    test = TestSceneReconstruction()
    test.setUpClass()

    # Perform test
    test.test_calibrated_corner_locations()

    # Generate new test data
    # test.save_csv()
