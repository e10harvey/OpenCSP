import os
import unittest

import numpy as np
import pytest

import opencsp.common.lib.deflectometry.ImageProjection as ip
import opencsp.common.lib.tool.exception_tools as et
import opencsp.common.lib.tool.file_tools as ft
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir


@pytest.mark.no_xvfb
class test_ImageProjection(unittest.TestCase):
    def setUp(self) -> None:
        path, _, _ = ft.path_components(__file__)
        self.data_dir = os.path.join(path, "data", "input", "ImageProjection")
        self.out_dir = os.path.join(path, "data", "output", "ImageProjection")
        self.file_image_projection_input = os.path.join(
            opencsp_code_dir(), 'test/data/sofast_common/image_projection_test.h5'
        )
        ft.create_directories_if_necessary(self.data_dir)
        ft.create_directories_if_necessary(self.out_dir)

    def tearDown(self):
        with et.ignored(Exception):
            ip.ImageProjection.instance().close()

    def test_set_image_projection(self):
        self.assertIsNone(ip.ImageProjection.instance())

        # Create a mock ImageProjection object
        image_projection = ip.ImageProjection.load_from_hdf(self.file_image_projection_input)

        # Test that the instance was set
        self.assertEqual(image_projection, ip.ImageProjection.instance())

        # Test un-setting the image_projection object
        image_projection.close()
        self.assertIsNone(ip.ImageProjection.instance())

    def test_on_close(self):
        global close_count
        close_count = 0

        def close_count_inc(image_projection):
            global close_count
            close_count += 1

        # Create a mock ImageProjection object with single on_close callback
        image_projection = ip.ImageProjection.load_from_hdf(self.file_image_projection_input)
        image_projection.on_close.append(close_count_inc)
        image_projection.close()
        self.assertEqual(close_count, 1)

        # Create a mock ImageProjection object with multiple on_close callback
        image_projection = ip.ImageProjection.load_from_hdf(self.file_image_projection_input)
        image_projection.on_close.append(close_count_inc)
        image_projection.on_close.append(close_count_inc)
        image_projection.close()
        self.assertEqual(close_count, 3)

        # Create a mock ImageProjection object without an on_close callback
        image_projection = ip.ImageProjection.load_from_hdf(self.file_image_projection_input)
        image_projection.close()
        self.assertEqual(close_count, 3)

    def test_zeros(self):
        # Create a mock ImageProjection object
        image_projection = ip.ImageProjection.load_from_hdf(self.file_image_projection_input)
        image_projection_data = ip.ImageProjectionData.load_from_hdf(self.file_image_projection_input)

        # Get the zeros array and verify its shape and values
        zeros = image_projection.get_black_array_active_area()
        self.assertEqual(
            (image_projection_data.active_area_size_y, image_projection_data.active_area_size_x, 3), zeros.shape
        )
        ones = zeros + 1
        self.assertEqual(
            np.sum(ones), image_projection_data.active_area_size_y * image_projection_data.active_area_size_x * 3
        )

    def test_to_from_hdf(self):
        # Load from HDF
        image_projection = ip.ImageProjection.load_from_hdf(self.file_image_projection_input)

        # Create output file
        file_h5_output = os.path.join(self.out_dir, "test_to_from_hdf.h5")

        # Save to HDF
        ft.delete_file(file_h5_output, error_on_not_exists=False)
        self.assertFalse(ft.file_exists(file_h5_output))
        image_projection.save_to_hdf(file_h5_output)
        self.assertTrue(ft.file_exists(file_h5_output))

        # Close, so that we don't have multiple windows open at a time
        image_projection.close()


if __name__ == '__main__':
    unittest.main()
