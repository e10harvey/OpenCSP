"""Unit test suite to test the camera_calibration library.

Change the boolean below to True and run to regenerate all test data
"""

from glob import glob
import os

import numpy as np
import cv2

import opencsp.app.camera_calibration.lib.calibration_camera as cc
import opencsp.app.camera_calibration.lib.image_processing as ip
import opencsp.app.sofast.lib.spatial_processing as sp
from opencsp.common.lib.tool.hdf5_tools import load_hdf5_datasets, save_hdf5_datasets


class TestCameraCalibration:
    @classmethod
    def setup_class(cls, regenerate=False):
        """Calculates camera calibration.

        Parameters
        ----------
        regenerate : bool, optional
            If true, saves data in output location instead of running unit tests, by default False
        """
        # Get test data location
        base_dir = os.path.join(os.path.dirname(__file__), 'data')

        # Pattern to search for captured images
        image_pattern = os.path.join(base_dir, 'images/*.png')
        test_data_file = os.path.join(base_dir, 'data_test.h5')

        # Define number of checkerboard corners
        npts = (18, 23)

        # Define camera name
        cam_name = 'Test Camera'

        # Find all files
        files = glob(image_pattern)
        files.sort()

        # Load images and find corners
        images = []
        p_object = []
        p_image = []
        for file in files:
            # Load image
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            p_cur_object, p_cur_image = ip.find_checkerboard_corners(npts, img)

            if p_cur_object is None or p_cur_image is None:
                continue

            # Save image, filename, and found corners
            images.append(img)

            p_object.append(p_cur_object)
            p_image.append(p_cur_image)

        # Calculate image size
        img_size = images[0].shape

        # Calibrate camera
        (camera, r_cam_object, v_cam_object_cam, calibration_error) = (
            cc.calibrate_camera(p_object, p_image, img_size, cam_name)
        )

        # Calculate reprojection errors
        errors = []
        for rot, vec, p_obj, p_img in zip(
            r_cam_object, v_cam_object_cam, p_object, p_image
        ):
            error = sp.reprojection_error(camera, p_obj, p_img, rot, vec)
            errors.append(error)

        # Save or load test data
        datasets = [
            'p_image_points',
            'p_object_points',
            'intrinsic_matrix',
            'distortion_coeffs',
            'image_shape_xy',
            'r_cam_object',
            'v_cam_object_cam',
            'calibration_error',
            'reprojection_errors',
        ]

        # Save calculated image and object points
        cls.p_image_points = np.array([a.data for a in p_image])
        cls.p_object_points = np.array([a.data for a in p_object])
        cls.intrinsic_matrix = camera.intrinsic_mat
        cls.distortion_coeffs = camera.distortion_coef
        cls.image_shape_xy = camera.image_shape_xy
        cls.r_cam_object = np.array([a.as_rotvec() for a in r_cam_object])
        cls.v_cam_object_cam = np.array([a.data for a in v_cam_object_cam]).squeeze()
        cls.calibration_error = calibration_error
        cls.reprojection_errors = np.array(errors)

        if regenerate:
            # Save test data in HDF file
            data = [
                cls.p_image_points,
                cls.p_object_points,
                cls.intrinsic_matrix,
                cls.distortion_coeffs,
                cls.image_shape_xy,
                cls.r_cam_object,
                cls.v_cam_object_cam,
                cls.calibration_error,
                cls.reprojection_errors,
            ]

            save_hdf5_datasets(data, datasets, test_data_file)
            print('Test data was created and saved to:', test_data_file)

        else:
            print('Performing tests...')

            # Load saved data
            data = load_hdf5_datasets(datasets, test_data_file)

            # Save expected data in class
            cls.p_image_points_exp = data['p_image_points']
            cls.Pxyz_object_points_exp = data['p_object_points']
            cls.intrinsic_matrix_exp = data['intrinsic_matrix']
            cls.distortion_coeffs_exp = data['distortion_coeffs']
            cls.image_shape_xy_exp = data['image_shape_xy']
            cls.R_cam_object_exp = data['r_cam_object']
            cls.V_cam_object_cam_exp = data['v_cam_object_cam']
            cls.calibration_error_exp = data['calibration_error']
            cls.reprojection_errors_exp = data['reprojection_errors']

            # idx = np.argmin(np.abs(cls.p_image_points - 1304.8412))
            idx = np.argmin(np.abs(cls.p_image_points - 1409.4342))
            print(idx)
            print(cls.p_image_points.flatten()[idx])

            loc = np.where(cls.p_image_points == cls.p_image_points.flatten()[idx])
            print(loc)

            print(cls.p_image_points.shape)

            # print([a.data[:, :2] for a in p_image[:2]])
            # print('\n')
            # print(np.array([a.data[:, :2] for a in p_image[:2]]))
            # print('\n')
            # print(cls.p_image_points[:2, :2, :2])
            # pts_test = cls.p_image_points[:2, :2, :2]
            # pts_test_exp = cls.p_image_points_exp[:2, :2, :2]
            # print(pts_test_exp)
            # print(pts_test)

    def test_image_points(self):
        np.testing.assert_allclose(self.p_image_points, self.p_image_points_exp)

    def test_object_points(self):
        np.testing.assert_allclose(self.p_object_points, self.Pxyz_object_points_exp)

    def test_intrinsic_matrix(self):
        np.testing.assert_allclose(self.intrinsic_matrix, self.intrinsic_matrix_exp)

    def test_distortion_coef(self):
        np.testing.assert_allclose(self.distortion_coeffs, self.distortion_coeffs_exp)

    def test_image_shape_xy(self):
        np.testing.assert_allclose(self.image_shape_xy, self.image_shape_xy_exp)

    def test_R_cam_object(self):
        np.testing.assert_allclose(self.r_cam_object, self.R_cam_object_exp)

    def test_V_cam_object(self):
        np.testing.assert_allclose(self.v_cam_object_cam, self.V_cam_object_cam_exp)

    def test_calibration_error(self):
        np.testing.assert_allclose(self.calibration_error, self.calibration_error_exp)

    def reprojection_errors(self):
        np.testing.assert_allclose(
            self.reprojection_errors, self.reprojection_errors_exp
        )


if __name__ == "__main__":
    # Set below boolean to True to save and overwrite new test data
    regen = False
    TestCameraCalibration.setup_class(regenerate=regen)
