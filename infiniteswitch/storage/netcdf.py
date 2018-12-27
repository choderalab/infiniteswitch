#!/usr/bin/env python

"""
Storage codec classes for NetCDF format.
"""

# ==============================================================================
# Global imports.
# ==============================================================================

import os
import weakref

import netCDF4 as netcdf

from infiniteswitch.storage import Codec


# ==============================================================================
# NetCDF4 array codec implementation.
# ==============================================================================

class _NetCDFArrayVariableView:
    """View into a NetCDF variable.

    This allows reading only slices from netcdf variables, and it emulates a
    numpy array API.

    """

    def __init__(self, netcdf_variable, is_appendable):
        self._netcdf_variable = netcdf_variable
        self._is_appendable = is_appendable

    def append(self, value):
        if not self._is_appendable:
            raise RuntimeError('This NetCDFArray is not appendable.')
        self[len(self._netcdf_variable)] = value

    def transpose(self):
        return self._netcdf_variable[:].transpose()

    def __len__(self):
        return self._netcdf_variable

    def __getitem__(self, item):
        return self._netcdf_variable[item]

    def __setitem__(self, item, value):
        self._netcdf_variable[item] = value

    def __iter__(self):
        return iter(self._netcdf_variable[:])

    def __eq__(self, other):
        return self._netcdf_variable == other

    def __copy__(self):
        """If copied, we don't need to keep track of the netcdf variable anymore."""
        return self._netcdf_variable[:]

    def __deepcopy__(self, memo):
        """If copied, we don't need to keep track of the netcdf variable anymore."""
        return self.__copy__()


class NetCDFArray(Codec):
    """Numeric array saved in netCDF format."""

    def __init__(self, relative_file_path, variable_path, datatype, dimensions,
                 is_appendable=False, **netcdf_variable_kwargs):
        super().__init__()
        self._relative_file_path = relative_file_path
        self._netcdf_variable_path = variable_path
        self._datatype = datatype
        self._dimensions = dimensions
        self._is_appendable = is_appendable
        self._netcdf_variable_kwargs = netcdf_variable_kwargs

    def get_file_path(self, storage):
        """The file path to the dataset associated to this codec."""
        return os.path.join(storage.directory_path, self._relative_file_path)

    def initialize(self, storage):
        """Initialize the storage data and error check."""
        # If this is an appendable variable, check that the first dimension is indefinite.
        if self._is_appendable and self._dimensions[0][1] != 0:
            raise ValueError('The first dimension of appendable variable must be 0 (i.e. indefinite).')

        # Check that all appendable variables of a dataset have different dimension
        # names for the first dimension. This restriction makes it straightforward
        # to append since the shape of the netcdf variable won't be affected by
        # appending values to other variables sharing the dimension.
        appendable_dimension_names = []
        for codec in storage.get_codecs():
            if (isinstance(codec, NetCDFArray) and codec._is_appendable and
                        codec.get_file_path(storage) == self.get_file_path(storage)):
                appendable_dimension_names.append(codec._dimensions[0][0])
        if len(appendable_dimension_names) != len(set(appendable_dimension_names)):
            raise ValueError('Appendable arrays must have different names for the first dimension.')

        # Initialize the class data if it wasn't already.
        shared_data = self.get_class_data(storage, default=None)
        if shared_data is None:
            # Map the Dataset objects by file path.
            self.set_class_data(storage, {'datasets': {}})

    def sync(self, storage):
        """Sync the netCDF data."""
        dataset = self._get_netcdf_dataset(storage, create=False)
        if dataset is not None and dataset.isopen():
            dataset.sync()

    def finalize(self, storage):
        """Close the NetCDF file."""
        dataset = self._get_netcdf_dataset(storage, create=False)
        if dataset is not None and dataset.isopen():
            dataset.close()

    def __get__(self, storage, storage_cls):
        variable = self._get_netcdf_variable(storage)
        return _NetCDFArrayVariableView(variable, is_appendable=self._is_appendable)

    def __set__(self, storage, new_value):
        # This codec simply stores the data in memory.
        variable = self._get_netcdf_variable(storage)
        new_value = self._validate(new_value)
        variable[:] = new_value

    # ------------------------------------ #
    # Retrieve netcdf dataset and variable #
    # ------------------------------------ #

    def _get_netcdf_dataset(self, storage, create=True):
        """Retrieve the NetCDF Dataset object from the storage and create one if requested.

        If create is False and the dataset doesn't exist, None is returned.

        We share the dataset instance with all other NetCDFArray instances in
        the storage class data.
        """
        datasets = self.get_class_data(storage)['datasets']

        # Retrieve the dataset object at this path.
        file_path = self.get_file_path(storage)
        try:
            dataset = datasets[file_path]
        except KeyError:
            # Create the Dataset object if requested.
            if create:
                # NetCDF can't create a new file for append.
                if storage.open_mode == 'a' and not os.path.isfile(file_path):
                    open_mode = 'w'
                else:
                    open_mode = storage.open_mode
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                dataset = netcdf.Dataset(file_path, open_mode)

                # Update the class data.
                datasets[file_path] = dataset
            else:
                dataset = None
        return dataset

    def _get_netcdf_variable(self, storage):
        """Retrieve the NetCDF variable object associated to this codec and create it if necessary."""
        dataset = self._get_netcdf_dataset(storage)

        # Resolve the NetCDF groups and variable name.
        group_path = self._netcdf_variable_path.split('/')
        group = dataset
        for group_name in group_path[:-1]:
            # Create the group if necessary.
            if group_name not in group.groups:
                group.createGroup(group_name)
            group = group.groups[group_name]
        variable_name = group_path[-1]

        # Create variable if not created yet.
        if variable_name in group.variables:
            variable = group.variables[variable_name]
        else:
            variable = self._create_netcdf_variable(dataset, group, variable_name)
        return variable

    def _create_netcdf_variable(self, dataset, group, variable_name):
        """Create a new variable with associated dimensions."""
        # Create the dimensions.
        for dimension_name, dimension_value in self._dimensions:
            if dimension_name not in dataset.dimensions:
                dataset.createDimension(dimension_name, dimension_value)

        # Create variable.
        dimension_names = tuple(k for k, v in self._dimensions)
        variable = group.createVariable(variable_name, self._datatype, dimension_names,
                                        **self._netcdf_variable_kwargs)
        return variable

