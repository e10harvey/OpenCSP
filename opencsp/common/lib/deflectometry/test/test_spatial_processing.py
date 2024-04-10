"""Unit test suite to test the spatial_processing library
"""
import os
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.common.lib.deflectometry.Display import Display
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe as Measurement
import opencsp.common.lib.deflectometry.spatial_processing as sp
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
from opencsp.common.lib.tool.hdf5_tools import load_hdf5_datasets


class TestSpatialProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get test data location
        base_dir = os.path.join(
            opencsp_code_dir(), 'test/data/sofast_measurements'
        )

        # Define test data files for single facet processing
        cls.data_file_facet = os.path.join(base_dir, 'calculations_facet/data.h5')
        cls.data_file_measurement = os.path.join(base_dir, 'measurement_facet.h5')

        cls.display = Display.load_from_hdf(cls.data_file_facet)
        cls.camera = Camera.load_from_hdf(cls.data_file_facet)

    def test_t_from_distance(self):
        datasets = [
            'DataSofastCalculation/image_processing/general/v_mask_centroid_image',
            'DataSofastCalculation/geometry/general/v_cam_optic_cam_exp',
        ]

        # Load test data
        data = load_hdf5_datasets(datasets, self.data_file_facet)
        measurement = Measurement.load_from_hdf(self.data_file_measurement)

        # Perform calculation
        v_cam_optic_cam_exp = sp.t_from_distance(
            Vxy(data['v_mask_centroid_image']),
            measurement.optic_screen_dist,
            self.camera,
            self.display.v_cam_screen_cam,
        ).data.squeeze()

        # Test
        np.testing.assert_allclose(data['v_cam_optic_cam_exp'], v_cam_optic_cam_exp)

    def test_r_from_position(self):
        datasets = [
            'DataSofastCalculation/geometry/general/v_cam_optic_cam_exp',
            'DataSofastCalculation/geometry/general/r_optic_cam_exp',
        ]

        # Load test data
        data = load_hdf5_datasets(datasets, self.data_file_facet)

        # Perform calculation
        r_optic_cam_exp = (
            sp.r_from_position(
                Vxyz(data['v_cam_optic_cam_exp']), self.display.v_cam_screen_cam
            )
            .inv()
            .as_rotvec()
        )

        # Test
        np.testing.assert_allclose(data['r_optic_cam_exp'], r_optic_cam_exp)

    def test_calc_rt_from_img_pts(self):
        datasets = [
            'DataSofastCalculation/geometry/general/r_optic_cam_exp',
            'DataSofastCalculation/geometry/general/v_cam_optic_cam_exp',
            'DataSofastCalculation/geometry/general/r_optic_cam_refine_1',
            'DataSofastCalculation/geometry/general/v_cam_optic_cam_refine_1',
            'DataSofastCalculation/image_processing/facet_000/loop_facet_image_refine',
            'DataSofastInput/optic_definition/facet_000/v_facet_corners',
        ]

        # Load test data
        data = load_hdf5_datasets(datasets, self.data_file_facet)

        # Perform calculation
        r_optic_cam, v_cam_optic_cam_refine = sp.calc_rt_from_img_pts(
            Vxy(data['loop_facet_image_refine']),
            Vxyz(data['v_facet_corners']),
            self.camera,
        )

        # Test
        np.testing.assert_allclose(
            data['r_optic_cam_refine_1'], r_optic_cam.as_rotvec(), atol=1e-5, rtol=0
        )
        np.testing.assert_allclose(
            data['v_cam_optic_cam_refine_1'],
            v_cam_optic_cam_refine.data.squeeze(),
            atol=1e-5,
            rtol=0,
        )

    def test_distance_error(self):
        datasets = [
            'DataSofastCalculation/error/error_optic_screen_dist_2',
            'DataSofastCalculation/geometry/general/v_cam_optic_cam_refine_2',
        ]

        # Load test data
        data = load_hdf5_datasets(datasets, self.data_file_facet)
        measurement = Measurement.load_from_hdf(self.data_file_measurement)

        # Perform calculation
        error_optic_screen_dist_2 = sp.distance_error(
            self.display.v_cam_screen_cam,
            Vxyz(data['v_cam_optic_cam_refine_2']),
            measurement.optic_screen_dist,
        )

        # Test
        np.testing.assert_allclose(
            data['error_optic_screen_dist_2'], error_optic_screen_dist_2
        )

    def test_reprojection_error(self):
        datasets = [
            'DataSofastCalculation/error/error_reprojection_2',
            'DataSofastCalculation/image_processing/facet_000/loop_facet_image_refine',
            'DataSofastCalculation/geometry/general/r_optic_cam_refine_1',
            'DataSofastCalculation/geometry/general/v_cam_optic_cam_refine_2',
            'DataSofastInput/optic_definition/facet_000/v_facet_corners',
        ]

        # Load test data
        data = load_hdf5_datasets(datasets, self.data_file_facet)

        # Perform calculation
        error_reproj_init = sp.reprojection_error(
            self.camera,
            Vxyz(data['v_facet_corners']),
            Vxy(data['loop_facet_image_refine']),
            Rotation.from_rotvec(data['r_optic_cam_refine_1']),
            Vxyz(data['v_cam_optic_cam_refine_2']),
        )

        # Test
        np.testing.assert_allclose(data['error_reprojection_2'], error_reproj_init)

    def test_refine_v_distance(self):
        datasets = [
            'DataSofastCalculation/geometry/general/v_cam_optic_cam_refine_1',
            'DataSofastCalculation/geometry/general/v_cam_optic_cam_refine_2',
            'DataSofastCalculation/geometry/general/r_optic_cam_refine_1',
        ]

        # Load test data
        data = load_hdf5_datasets(datasets, self.data_file_facet)
        measurement = Measurement.load_from_hdf(self.data_file_measurement)

        # Perform calculation
        r_cam_optic = Rotation.from_rotvec(data['r_optic_cam_refine_1'])
        v_meas_pt_optic_cam = measurement.measure_point.rotate(r_cam_optic)
        v_cam_optic_cam_refine_2 = sp.refine_v_distance(
            Vxyz(data['v_cam_optic_cam_refine_1']),
            measurement.optic_screen_dist,
            self.display.v_cam_screen_cam,
            v_meas_pt_optic_cam,
        ).data.squeeze()

        # Test
        np.testing.assert_allclose(
            data['v_cam_optic_cam_refine_2'], v_cam_optic_cam_refine_2
        )
