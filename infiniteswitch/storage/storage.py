#!/usr/bin/env python

"""
Base classes to create storage schemas and objects.
"""

# ==============================================================================
# Global imports.
# ==============================================================================

import copy
import inspect
import os


# ==============================================================================
# General storage class.
# ==============================================================================

class Storage:
    """Helper class used to store and read data from disk.

    The class store the list of codecs, the path to a base directory,
    and a weak reference dictionary mapping file paths to their file
    object (if any) that can be used to coordinate codecs.

    """

    def __init__(self, directory_path, open_mode='r'):
        os.makedirs(directory_path, exist_ok=True)
        self._directory_path = directory_path
        self._open_mode = open_mode
        # Data that is specific to a codec instance.
        self._codec_instance_data = {}
        # Data that is shared among all instances of a codec
        # class that is assigned to this storage instance.
        self._codec_class_data = {}

        # Initialize all codecs.
        self._initialize()

    @property
    def directory_path(self):
        """The main directory where data is stored (read-only)."""
        return self._directory_path

    @property
    def open_mode(self):
        """The open mode of the file (read-only)."""
        return self._open_mode

    def _initialize(self):
        """Initialize all codecs.

        This enable the codec objects to perform error checking and any
        operation required before start reading/writing.
        """
        for codec in self.get_codecs():
            codec.initialize(self)

    def sync(self):
        """Force synchronization of all data with disk."""
        for codec in self.get_codecs():
            codec.sync(self)

    def close(self):
        """Finalize and close the file(s)."""
        self.sync()
        for codec in self.get_codecs():
            codec.finalize(self)

    @classmethod
    def get_codecs(cls):
        """Retrieve the list of codecs associated to this storage class."""
        return [c for name, c in inspect.getmembers(cls) if isinstance(c, Codec)]

    def __del__(self):
        """Always finalize before destroying the object."""
        self.close()


# ==============================================================================
# General codec class.
# ==============================================================================

class Codec:
    """Base class for descriptors encoding and decoding data.

    All codecs must inherit from this base class. This basic
    implementation simply keeps the data in memory.
    """

    def __init__(self, _validator=None):
        # _validator should be assigned with the decorator (e.g. @codec.validator).
        self._validator = _validator

    def initialize(self, storage):
        """Initialize the codec."""
        # No need to initialize anything since data is always in memory.
        pass

    def sync(self, storage):
        """Force synchronization to disk."""
        # No need to sync since data is always in memory.
        pass

    def finalize(self, storage):
        # No need to close any file since data is always in memory.
        pass

    def __get__(self, storage, storage_cls):
        # This codec simply stores the data in memory.
        return self.get_instance_data(storage)

    def __set__(self, storage, new_value):
        # This codec simply stores the data in memory.
        new_value = self._validate(new_value)
        self.set_instance_data(storage, new_value)

    # ------------------------------ #
    # Methods for inheriting objects #
    # ------------------------------ #

    def __set_name__(self, storage_cls, name):
        """Automatically sets the name of the storage attribute."""
        self._name = name

    def get_instance_data(self, storage, default=None):
        """Retrieve the data associated to this codec instance."""
        return storage._codec_instance_data.get(self._name, default)

    def set_instance_data(self, storage, new_value):
        """Set the data associated to this codec instance."""
        storage._codec_instance_data[self._name] = new_value

    @classmethod
    def get_class_data(cls, storage, default=None):
        """Retrieve the data associated to all instances of this codec class from the storage."""
        return storage._codec_class_data.get(cls, default)

    @classmethod
    def set_class_data(cls, storage, new_value):
        """Set the data associated to all instances of this codec class in the storage."""
        storage._codec_class_data[cls] = new_value

    def validator(self, validator):
        """Enable expressing validators with the usual property syntax for getter/setters."""
        codec = copy.deepcopy(self)
        codec._validator = validator
        return codec

    def _validate(self, new_value):
        """Convenience function to validate a value when a validator is specified."""
        if self._validator is None:
            return new_value
        return self._validator(new_value)
