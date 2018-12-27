"""
Infinite Switch Hamiltonian Exchange
Implementation and testing of the InfiniteSwitchIntegrator.
"""

# Add imports here
from .infiniteswitch import *
from .storage import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
