#!/usr/bin/env python

"""
Unit and regression test for the storage module.
"""

# ==============================================================================
# Global imports.
# ==============================================================================

import os
import shutil
import tempfile

import numpy as np
import nose

from infiniteswitch.storage import Storage, NetCDFArray


# ==============================================================================
# Tests for NetCDF codecs
# ==============================================================================

class TestNetCDFArray:
    """Test the NetCDF array codec."""

    # Storage class to test.
    class Store(Storage):
        var1 = NetCDFArray(
            relative_file_path = 'dataset.nc',
            variable_path='var1',
            datatype='f8',
            is_appendable=True,
            dimensions=[('iteration', 0), ('replicas1', 2), ('states1', 5)]
        )
        # Try different file, indefinite dimension only.
        var2 = NetCDFArray(
            relative_file_path = 'nested_path/dataset2.nc',
            variable_path='var2',
            datatype='i4',
            is_appendable=True,
            dimensions=[('iteration', 0)]
        )
        # Try same file, but with a group and different datatype and without indefinite dimension.
        var3 = NetCDFArray(
            relative_file_path = 'dataset.nc',
            variable_path='mygroup/var3',
            datatype='i4',
            dimensions=[('replicas2', 1), ('states2', 3)]
        )

    @classmethod
    def setup_class(cls):
        """Common setup."""
        cls.storage_dir_path = tempfile.mkdtemp()

        # Open a storage for writing.
        write_storage = cls.Store(directory_path=cls.storage_dir_path, open_mode='w')

        # Push to indefinite-dimensional arrays with normal setting and append.
        var1_element = np.ones((2, 5))
        write_storage.var1[0] = var1_element
        write_storage.var1.append(var1_element)
        cls.expected_var1 = np.array([var1_element, var1_element])

        # Same for var 1 but modify the original assignment.
        write_storage.var2.append(3)
        write_storage.var2[1] = 3
        write_storage.var2[0] = 2
        cls.expected_var2 = np.array([2, 3])

        # Write value and modify it for var2.
        cls.expected_var3 = np.zeros((1, 3))
        write_storage.var3 = np.random.randn(*cls.expected_var3.shape)
        write_storage.var3 = cls.expected_var3

        write_storage.close()

    @classmethod
    def teardown_class(cls):
        """Remove temporary directory."""
        shutil.rmtree(cls.storage_dir_path)

    def test_separate_files(self):
        """Test that writing creates two separate files."""
        assert os.path.isfile(os.path.join(self.storage_dir_path, 'dataset.nc'))
        assert os.path.isfile(os.path.join(self.storage_dir_path, 'nested_path/dataset2.nc'))

    def test_reading(self):
        """The information is restored correctly."""
        read_storage = self.Store(self.storage_dir_path, open_mode='r')
        assert np.all(read_storage.var1 == self.expected_var1), (read_storage.var1[:], self.expected_var1)
        assert np.all(read_storage.var2 == self.expected_var2)
        assert np.all(read_storage.var3 == self.expected_var3)

    def test_groups(self):
        """Check that groups are handled correctly."""
        from netCDF4 import Dataset
        dataset = Dataset(os.path.join(self.storage_dir_path, 'dataset.nc'), 'r')
        assert 'mygroup' in dataset.groups
        assert 'var3' in dataset.groups['mygroup'].variables
        dataset.close()

    @staticmethod
    def test_appendable_dimension_name():
        """Test that an error is raised with appendable dimension name conflicts."""
        class Schema(Storage):
            var1 = NetCDFArray(
                relative_file_path = 'dataset.nc',
                variable_path='var1',
                datatype='f8',
                is_appendable=True,
                dimensions=[('iteration', 0)]
            )
            var2 = NetCDFArray(
                relative_file_path = 'dataset.nc',
                variable_path='var2',
                datatype='f8',
                is_appendable=True,
                dimensions=[('iteration', 0)]
            )
        with tempfile.TemporaryDirectory() as tmp_dir_path:
            with nose.tools.assert_raises_regexp(ValueError, 'different names'):
                Schema(tmp_dir_path, open_mode='w')


