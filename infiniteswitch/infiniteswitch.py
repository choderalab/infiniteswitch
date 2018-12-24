#!/usr/bin/env python

"""
infiniteswitch.py

Implementation of the InfiniteSwitchIntegrator for the infinite switch
Hamiltonian exchange algorithm in OpenMM.

"""

# ==============================================================================
# Global imports.
# ==============================================================================

import collections
import copy
from typing import List

import numpy as np
import openmmtools as mmtools
from simtk import openmm


# ==============================================================================
# Utility functions to create paths and determine the path quadrature nodes.
# ==============================================================================

def get_linearly_interpolated_path(initial, final):
    """Construct a linear path between initial and final.

    The initial and final points are at alpha = -1 and 1 respectively
    to make Gauss-Legendre quadrature trivial to apply.

    Parameters
    ----------
    initial : float
        The value of the point at alpha = -1.
    final : float
        The value of the point at alpha = 1.

    Returns
    -------
    path : func
        A path with signature path(alpha)
    path_derivative_norm : func
        The norm of the path derivative with signature
        path_derivative_norm(alpha).
    """
    # Make sure initial and final points to np.arrays.
    if np.isscalar(initial):
        initial = [initial]
    if np.isscalar(final):
        final = [final]
    initial = np.array(initial)
    final = np.array(final)

    def _path(alpha):
        if alpha < 0:
            return (-alpha*initial + (1 + alpha)*(final + initial)/2).tolist()
        else:
            return ((1 - alpha)*(final + initial)/2 + alpha*final).tolist()

    def _path_derivative_norm(alpha):
        return float(np.linalg.norm((final - initial) / 2))

    return _path, _path_derivative_norm


def determine_quadrature(curve, curve_derivative_norm, deg):
    """Determine Gauss-Legendre nodes, weights, and gradient norms for a curve."""
    nodes, weights = np.polynomial.legendre.leggauss(deg=deg)
    # quadrature[force_group_idx] == [force group quadrature protocol]
    quadrature_path = [curve(node) for node in nodes]
    path_derivative = [curve_derivative_norm(node) for node in nodes]
    return quadrature_path, weights, path_derivative


# ==============================================================================
# Infinite switch Hamiltonian Exchange integrator.
# ==============================================================================

class InfiniteSwitchIntegrator(mmtools.integrators.LangevinIntegrator):
    """Integrator implementing the infinite switch Hamiltonian exchange sampling algorithm.

    Parameters
    ----------
    system : simtk.openmm.System
        The System that will be simulated in the infinite switch limit.
        The system's forces are redistributed among the force groups to optimize
        the calculation. You should not modify the system after initializing the
        integrator.
    quadrature_nodes: Dict[Union[str, int], List[float]]
        The quadrature nodes used to compute the integral. Each key of
        the dictionary maps the name of the parameter to a list of its values
        for all the quadrature nodes.
        If the name of the parameter is an integer, this is interpreted as the
        index of a force, which is assumed to be in Gibbs ensemble with respect
        to its parameter. The latter case, allows some optimizations that result
        in faster propagation than in the general case.
        If the force associated to the (string) parameter is a ``NonbondedForce``,
        the potential is linearly interpolated between the initial and final states.
    quadrature_nodes_gradient_norm : List[float]
        The norm of the derivative at each quadrature node.
    quadrature_weights: List[float]
        The weights of the quadrature scheme used to estimate the integral.
    *args
    ee_weight_stepsize : float, optional
        The step size for the expanded ensemble weight update. Set this to 0.0
        to prevent the online updating.
    **kwargs
        Parameters passed to super().

    """

    def __init__(self, system, quadrature_nodes, quadrature_nodes_gradient_norm,
                 quadrature_weights, *args, ee_weight_stepsize=1.0, **kwargs):

        # All the lists of parameters values in the quadrature_nodes must have the same length.
        protocol_lengths = {len(v) for k, v in quadrature_nodes.items()}
        if len(protocol_lengths) != 1:
            raise ValueError('quadrature_nodes must have one or more parameters '
                             'associated to a list of values of the same length.')
        # All the quadrature arguments must have the same length.
        protocol_lengths.update([len(quadrature_weights), len(quadrature_nodes_gradient_norm)])
        if len(protocol_lengths) != 1:
            raise ValueError('quadrature_nodes, quadrature_weights and '
                             'quadrature_nodes_gradient_norm must have the same length.')

        # Store integrator arguments that are necessary for the initialization
        # of the integrator instance. These attributes are lost after XML
        # serialization, but they are not necessary after initialization.
        self._gibbs_force_groups = []
        self._interpolated_force_groups = []
        self._parametric_force_groups = []
        self._nonparametric_force_groups = []
        # Optimize the system force groups. The new quadrature_nodes have the
        # Gibbs components organized by force groups rather than force indices.
        quadrature_nodes = self._optimize_force_groups(system, quadrature_nodes)

        self._quadrature_nodes_by_parameter = quadrature_nodes
        # We always use weights and derivatives together in
        # quadrature, so we might as well multiply them now.
        self._quadrature_weights = [w*d for w, d in zip(quadrature_weights, quadrature_nodes_gradient_norm)]

        self._ee_weight_stepsize = ee_weight_stepsize

        super().__init__(*args, **kwargs)

    # -------------------------------------------------------------------------
    # Constants to avoid hardcoding Lepton variable names.
    # -------------------------------------------------------------------------

    # Prefix to reduce the probability of global variable name collisions.
    MANGLE_PREFIX = '__infinite_switch_'

    # Flags to keep track of what has been invalidated.
    UPDATE_INTEGRAL_WEIGHTS_VARNAME = MANGLE_PREFIX + 'update_integral_weights'
    UPDATE_AVERAGE_FORCE_VARNAME = MANGLE_PREFIX + 'update_average_force'

    # Prefix of the potential energies for each force group and each node.
    # It's followed by the force group or quadrature node index respectively.
    NODE_ENERGY_PREFIX = MANGLE_PREFIX + 'node_energy_'
    FORCE_GROUP_ENERGY_PREFIX = MANGLE_PREFIX + 'force_group_energy_'
    MIN_NODE_ENERGY_VARNAME = MANGLE_PREFIX + 'min_node_energy'
    # For interpolated forces, we need both the initial and final value of the energy.
    INTERPOLATED_FORCE_GROUP_INITIAL_ENERGY_PREFIX = MANGLE_PREFIX + 'force_group_initial_energy_'
    INTERPOLATED_FORCE_GROUP_FINAL_ENERGY_PREFIX = MANGLE_PREFIX + 'force_group_final_energy_'

    # Global variables used to update the online estimates.
    N_STEPS_VARNAME = MANGLE_PREFIX + 'n_steps'
    EXPANDED_ENSEMBLE_WEIGHT_STEPSIZE_VARNAME = MANGLE_PREFIX + 'ee_weight_stepsize'
    # Expanded ensemble weight and partition functions. It's followed by the node index.
    NODE_EXPANDED_ENSEMBLE_WEIGHT_PREFIX = MANGLE_PREFIX + 'w_'
    NODE_PARTITION_FUNCTION_PREFIX = MANGLE_PREFIX + 'z_'
    EXPANDED_ENSEMBLE_WEIGHT_NORMALIZING_CONSTANT_VARNAME = MANGLE_PREFIX + 'w_normalizing_constant'

    # Integral weight prefix. It's followed by the quadrature node index.
    NODE_INTEGRAL_WEIGHT_PREFIX = MANGLE_PREFIX + 'integral_weight_'
    INTEGRAL_WEIGHT_SUM_VARNAME = MANGLE_PREFIX + 'integral_weight_sum'

    # Average parameters for calculation of Gibbs and interpolation forces.
    # It's followed by the force group index.
    AVERAGE_PARAMETER_PREFIX = MANGLE_PREFIX + 'average_parameter_'
    # Final value of the average force.
    AVERAGE_FORCE_VARNAME = MANGLE_PREFIX + 'average_force'

    # -------------------------------------------------------------------------
    # Public properties and methods.
    # -------------------------------------------------------------------------

    @property
    def ee_weight_stepsize(self):
        """The time step for the expanded ensemble weight update."""
        return self.getGlobalVariableByName(self.EXPANDED_ENSEMBLE_WEIGHT_STEPSIZE_VARNAME)

    @ee_weight_stepsize.setter
    def ee_weight_stepsize(self, new_value):
        self.setGlobalVariableByName(self.EXPANDED_ENSEMBLE_WEIGHT_STEPSIZE_VARNAME, new_value)

    def get_forces(self):
        """Return the average force."""
        return self.getPerDofVariableByName(self.AVERAGE_FORCE_VARNAME)

    def get_nodes_partition_functions(self):
        """Return the current estimates of the partition functions."""
        return [self.getGlobalVariableByName(self.NODE_PARTITION_FUNCTION_PREFIX + str(i))
                for i in range(self._n_quadrature_nodes)]

    def get_nodes_energies(self):
        """Retrieve the potential evaluated at each nodes.

        Only the terms that depend on the Hamiltonian parameters are included,
        as the nonparametric terms are not cached during the propagation.
        """
        node_energy_varnames = [self.NODE_ENERGY_PREFIX + str(i)
                                for i in range(self._n_quadrature_nodes)]
        return [self.getGlobalVariableByName(name) for name in node_energy_varnames]

    def get_nodes_expanded_ensemble_weights(self):
        """Retrieve the expanded ensemble weights of all quadrature nodes."""
        ee_weight_varnames = [self.NODE_EXPANDED_ENSEMBLE_WEIGHT_PREFIX + str(i)
                              for i in range(self._n_quadrature_nodes)]
        return [self.getGlobalVariableByName(name) for name in ee_weight_varnames]

    def get_nodes_integral_weights(self):
        """Retrieve the normalized integral weights of all quadrature nodes."""
        integral_weight_varnames = [self.NODE_INTEGRAL_WEIGHT_PREFIX + str(i)
                                    for i in range(self._n_quadrature_nodes)]
        return [self.getGlobalVariableByName(name) for name in integral_weight_varnames]

    # -------------------------------------------------------------------------
    # Inherited from LangevinIntegrator.
    # -------------------------------------------------------------------------

    def _add_global_variables(self):
        """Override the parent method to add all the InfiniteSwitchIntegrator variables."""
        super()._add_global_variables()

        # Variables to cache the energy of force groups.
        for energy_varname in self._get_force_group_energy_varnames().values():
            self.addGlobalVariable(energy_varname, 0.0)
        for initial_energy_varname, final_energy_varname in self._get_interpolated_force_group_energy_varnames().values():
            self.addGlobalVariable(initial_energy_varname, 0.0)
            self.addGlobalVariable(final_energy_varname, 0.0)

        # Variables to cache the energy of quadrature nodes and integral weights.
        self.addGlobalVariable(self.MIN_NODE_ENERGY_VARNAME, 0.0)
        self.addGlobalVariable(self.INTEGRAL_WEIGHT_SUM_VARNAME, 0.0)
        for node_idx in range(self._n_quadrature_nodes):
            self.addGlobalVariable(self.NODE_ENERGY_PREFIX + str(node_idx), 0.0)
            self.addGlobalVariable(self.NODE_INTEGRAL_WEIGHT_PREFIX + str(node_idx), 0.0)

        # Variables to cache the average forces.
        self.addPerDofVariable(self.AVERAGE_FORCE_VARNAME, 0.0)
        for force_group_idx in self._gibbs_force_groups + self._interpolated_force_groups:
            # Interpolated forces also have the parameter name.
            force_group_idx = force_group_idx if isinstance(force_group_idx, int) else force_group_idx[0]
            self.addGlobalVariable(self.AVERAGE_PARAMETER_PREFIX + str(force_group_idx), 0.0)

        # Flags for update.
        self.addGlobalVariable(self.UPDATE_INTEGRAL_WEIGHTS_VARNAME, 1.0)
        self.addGlobalVariable(self.UPDATE_AVERAGE_FORCE_VARNAME, 1.0)

        # Variable for online estimates.
        self.addGlobalVariable(self.N_STEPS_VARNAME, 0.0)
        self.addGlobalVariable(self.EXPANDED_ENSEMBLE_WEIGHT_STEPSIZE_VARNAME, self._ee_weight_stepsize)
        # Expanded ensemble weights must be normalized.
        expanded_ensemble_weight = 1 / sum(self._quadrature_weights)
        for node_idx in range(self._n_quadrature_nodes):
            self.addGlobalVariable(self.NODE_EXPANDED_ENSEMBLE_WEIGHT_PREFIX + str(node_idx), expanded_ensemble_weight)
            self.addGlobalVariable(self.NODE_PARTITION_FUNCTION_PREFIX + str(node_idx), 0.0)
        self.addGlobalVariable(self.EXPANDED_ENSEMBLE_WEIGHT_NORMALIZING_CONSTANT_VARNAME, 1.0)

    def _add_integrator_steps(self):
        super()._add_integrator_steps()
        # Update estimates of the partition functions and expanded ensemble weights.
        self._add_update_online_estimates()

    def _add_V_step(self, force_group="0"):
        """Deterministic velocity update, using only forces from force-group fg.

        Parameters
        ----------
        force_group : str, optional, default="0"
           Force group to use for this step
        """
        if self._measure_shadow_work:
            self.addComputeSum("old_ke", self._kinetic_energy)

        if self._mts:
            raise ValueError('Cannot support MTS with infinite switch.')
        if force_group != '':
            raise ValueError('Cannot support multiple force groups: {}.'.format(force_group))

        # Compute the force of the marginal distribution.
        self._add_compute_average_force()

        # Update velocities.
        self.addComputePerDof("v", "v + (dt / {}) * {} / m".format(self._force_group_nV["0"],
                                                                   self.AVERAGE_FORCE_VARNAME))
        self.addConstrainVelocities()

        if self._measure_shadow_work:
            self.addComputeSum("new_ke", self._kinetic_energy)
            self.addComputeGlobal("shadow_work", "shadow_work + (new_ke - old_ke)")

    def _add_R_step(self):
        super()._add_R_step()
        self.addComputeGlobal(self.UPDATE_INTEGRAL_WEIGHTS_VARNAME, '1.0')
        self.addComputeGlobal(self.UPDATE_AVERAGE_FORCE_VARNAME, '1.0')

    # -------------------------------------------------------------------------
    # Convenience properties and functions.
    # -------------------------------------------------------------------------

    @property
    def _n_quadrature_nodes(self):
        """The number of quadrature nodes."""
        # _quadrature_weights is lost after serialization, but this variable
        # is used inside public methods so here we retrieve this information
        # from the serialization if needed.
        if hasattr(self, '_quadrature_weights'):
            self._cached_n_quadrature_nodes = len(self._quadrature_weights)
        elif not hasattr(self, '_cached_n_quadrature_nodes'):
            node_idx = 0
            while True:
                try:
                    self.getGlobalVariableByName(self.NODE_INTEGRAL_WEIGHT_PREFIX + str(node_idx))
                except:
                    break
                node_idx += 1
            self._cached_n_quadrature_nodes = node_idx
        return self._cached_n_quadrature_nodes

    @property
    def _quadrature_nodes_by_node(self):
        """Returns self._quadrature_nodes as a list of dictionaries rather than a dictionary of lists."""
        return [{k: v[i] for k, v in self._quadrature_nodes_by_parameter.items()}
                for i in range(self._n_quadrature_nodes)]

    def _get_interpolating_parameter(self, parameter_name, node_idx):
        """Compute the value of the interpolating parameter at the given node."""
        # The offset parameter affects the charges so we square the parameter values.
        initial_parameter_value = self._quadrature_nodes_by_parameter[parameter_name][0]**2
        final_parameter_value = self._quadrature_nodes_by_parameter[parameter_name][-1]**2
        parameter_value = self._quadrature_nodes_by_parameter[parameter_name][node_idx]**2
        return (parameter_value - final_parameter_value) / (initial_parameter_value - final_parameter_value)

    def _get_force_group_energy_varnames(self, only_gibbs=False, only_parametric=False):
        """The Lepton variable names for all force group energies (except interpolated)."""
        energy_varnames = collections.OrderedDict()
        parametric_force_group_indices = [i for i, parameter_name in self._parametric_force_groups]
        if only_gibbs:
            force_group_indices = self._gibbs_force_groups
        elif only_parametric:
            force_group_indices = parametric_force_group_indices
        else:
            force_group_indices = self._gibbs_force_groups + parametric_force_group_indices + self._nonparametric_force_groups
        for force_group_idx in force_group_indices:
            energy_varnames[force_group_idx] = self.FORCE_GROUP_ENERGY_PREFIX + str(force_group_idx)
        return energy_varnames

    def _get_interpolated_force_group_energy_varnames(self):
        """The Lepton variable names for all the interpolated force group energies."""
        prefixes = [self.INTERPOLATED_FORCE_GROUP_INITIAL_ENERGY_PREFIX,
                    self.INTERPOLATED_FORCE_GROUP_FINAL_ENERGY_PREFIX]
        energy_varnames = collections.OrderedDict()
        for force_group_idx, parameter_name in self._interpolated_force_groups:
            energy_varnames[force_group_idx] = [prefix + str(force_group_idx) for prefix in prefixes]
        return energy_varnames

    # -------------------------------------------------------------------------
    # Initialization.
    # -------------------------------------------------------------------------

    def _optimize_force_groups(self, system, quadrature_nodes):
        """Separate the forces into different force groups to allow maximum efficiency.

        After force group optimization, the method updates the following attribute:

            self._nonparametric_force_groups
            self._parametric_force_groups
            self._gibbs_force_groups
            self._interpolated_force_groups

        Returns
        -------
        quadrature_nodes
            A new quadrature node list in which the indices of the forces in Gibbs
            ensemble are replaced by the indices of the force groups in Gibbs ensemble.
            Everything else (i.e. the protocol for the global parameters) is the same.

        Raises
        ------
        ValueError
            If a Gibb force is controlled by a generic Hamiltonian parameter, if a
            force is controlled by more than one Hamiltonian parameter, or if not
            all Hamiltonian parameters specified in quadrature_nodes can be found.
        """
        # TODO: Split reciprocal space into its own group with PME and offset parameters affecting charges.

        # The indices of the forces in Gibbs ensemble are all the numeric keys of the quadrature_nodes.
        gibbs_force_indices = sorted(k for k in quadrature_nodes if isinstance(k, int))
        interpolated_force_indices = []
        parametric_force_indices = []
        nonparametric_force_indices = []

        # Classify forces.
        # ----------------

        # Divide the remaining forces depending on whether they expose
        # an Hamiltonian parameter that is exchanged or not.
        hamiltonian_parameters = {k for k in quadrature_nodes if isinstance(k, str)}
        parameter_to_force_indices = {k: [] for k in hamiltonian_parameters}
        interpolated_parameter_to_force_indices = {k: [] for k in hamiltonian_parameters}
        found_parameters = set()
        for force_idx, force in enumerate(system.getForces()):
            # If this force doesn't define global parameters, we classify it as nonparametric.
            try:
                n_force_parameters = force.getNumGlobalParameters()
            except AttributeError:
                force_quadrature_parameters = set()
            else:
                # Check if any of the force's global parameters is modified in the marginal integral.
                force_parameters = {force.getGlobalParameterName(i) for i in range(n_force_parameters)}
                force_quadrature_parameters = force_parameters & hamiltonian_parameters

            # If the Gibb force can't be controlled by general Hamiltonian parameters.
            if len(force_quadrature_parameters) > 0 and force_idx in gibbs_force_indices:
                raise ValueError('Force {} both in Gibbs ensemble and controlled by the '
                                 'Hamiltonian parameters {}'.format(force, force_quadrature_parameters))

            # Classify this as nonparametric if there are no global parameters that are exchanged.
            if len(force_quadrature_parameters) == 0:
                if force_idx not in gibbs_force_indices:
                    nonparametric_force_indices.append(force_idx)
                continue

            # We don't currently support forces controlled by multiple Hamiltonian parameters.
            # It makes it a pain to optimize the calculation of energies and forces when they
            # don't change from one quadrature node to another.
            if len(force_quadrature_parameters) > 1:
                raise ValueError('Force {} is controlled by multiple Hamiltonian parameters: '
                                 '{}'.format(force, force_quadrature_parameters))
            force_quadrature_parameter = list(force_quadrature_parameters)[0]

            # If this is a NonbondedForce, we interpolate the energy between the extremes.
            if isinstance(force, openmm.NonbondedForce):
                interpolated_force_indices.append(force_idx)
                interpolated_parameter_to_force_indices[force_quadrature_parameter].append(force_idx)
            else:
                # Otherwise, classify this as a parametric force groups.
                parametric_force_indices.append(force_idx)
                # Update the map from the parameter name to the force indices.
                parameter_to_force_indices[force_quadrature_parameter].append(force_idx)
            # Update the found parameters for error checking later.
            found_parameters.add(force_quadrature_parameter)

        # Raise an error if we haven't found all parameters.
        if len(found_parameters) != len(hamiltonian_parameters):
            raise ValueError('Cannot find some of the Hamiltonian parameters as global parameters.')

        # Optimally split forces into force groups.
        # -----------------------------------------
        current_force_group = -1

        # Put Gibbs forces that have the same protocol in the same force groups.
        protocol_to_forces = collections.OrderedDict()
        # Don't modify the original variable.
        quadrature_nodes = copy.deepcopy(quadrature_nodes)
        for force_idx in gibbs_force_indices:
            # We pop the Gibb force protocol that will be added grouped by force group later.
            force_protocol = tuple(quadrature_nodes.pop(force_idx))
            try:
                protocol_to_forces[force_protocol].append(force_idx)
            except KeyError:
                protocol_to_forces[force_protocol] = [force_idx]
        # Assign force groups to Gibbs forces.
        for force_protocol, force_indices in protocol_to_forces.items():
            current_force_group += 1
            for force_idx in force_indices:
                system.getForce(force_idx).setForceGroup(current_force_group)
            self._gibbs_force_groups.append(current_force_group)
            # Modify quadrature_nodes Gibbs protocols to be grouped by force groups instead of force indices.
            quadrature_nodes[current_force_group] = list(force_protocol)

        def split_forces_by_parameter(par_to_force_indices, force_groups):
            nonlocal current_force_group
            # Sort alphabetically so that the serialization of the integrator is deterministic
            for parameter_name in sorted(par_to_force_indices):
                # Skip if there are no forces associated to this parameter.
                if len(par_to_force_indices[parameter_name]) == 0:
                    continue
                current_force_group += 1
                for force_idx in par_to_force_indices[parameter_name]:
                    system.getForce(force_idx).setForceGroup(current_force_group)
                force_groups.append((current_force_group, parameter_name))

        # Put interpolated forces that are controlled by the same parameter into the same force group.
        split_forces_by_parameter(interpolated_parameter_to_force_indices, self._interpolated_force_groups)
        # Put all parametric forces that are controlled by the same parameter into the same force group.
        split_forces_by_parameter(parameter_to_force_indices, self._parametric_force_groups)

        # Put all nonparametric forces into a single force group.
        current_force_group += 1
        for force_idx in nonparametric_force_indices:
            system.getForce(force_idx).setForceGroup(current_force_group)
        self._nonparametric_force_groups.append(current_force_group)

        return quadrature_nodes

    # -------------------------------------------------------------------------
    # Add compute functions.
    # -------------------------------------------------------------------------

    def _add_compute_average_force(self):
        self.beginIfBlock(self.UPDATE_AVERAGE_FORCE_VARNAME + ' > 0')

        self._add_compute_integral_weights()

        initialized = False
        def cumulate_average_force(force_component_expression):
            nonlocal initialized
            if initialized:
                expression = self.AVERAGE_FORCE_VARNAME +  ' + ' + force_component_expression
            else:
                expression = force_component_expression
                initialized = True
            self.addComputePerDof(self.AVERAGE_FORCE_VARNAME, expression)

        # Nonparametric forces contribution.
        for force_group_idx in self._nonparametric_force_groups:
            cumulate_average_force('f' + str(force_group_idx))

        # Gibbs and interpolated forces contribution.
        self._add_compute_average_parameters()

        for force_group_idx in self._gibbs_force_groups:
            average_parameter_varname = self.AVERAGE_PARAMETER_PREFIX + str(force_group_idx)
            cumulate_average_force(average_parameter_varname + ' * f' + str(force_group_idx))

        for force_group_idx, parameter_name in self._interpolated_force_groups:
            for parameter_value_idx, interpolating_expression in [[0, '{alpha}'], [-1, '(1 - {alpha})']]:
                parameter_value = self._quadrature_nodes_by_parameter[parameter_name][parameter_value_idx]
                self.addComputeGlobal(parameter_name, str(parameter_value))
                force_expression = interpolating_expression + ' * f' + str(force_group_idx)
                cumulate_average_force(force_expression.format(alpha=self.AVERAGE_PARAMETER_PREFIX+str(force_group_idx)))

        # Parametric forces contributions.
        integrand_expression = 'f{0} * ' + self.NODE_INTEGRAL_WEIGHT_PREFIX + '{1}'
        for force_group_idx, parameter_name in self._parametric_force_groups:
            for node_idx, parameter_value in enumerate(self._quadrature_nodes_by_parameter[parameter_name]):
                # Set the value of the parameter before computing the force.
                self.addComputeGlobal(parameter_name, str(parameter_value))
                cumulate_average_force(integrand_expression.format(force_group_idx, node_idx))

        self.addComputeGlobal(self.UPDATE_AVERAGE_FORCE_VARNAME, '0.0')
        self.endBlock()

    def _add_compute_integral_weights(self):
        """Compute all the integral weights and their normalizing constant.

        The integral weight is given by

            B_i * dphi * e^{-beta*V(q, lambda)} * w

        where B_i is the quadrature weight, and dphi is the derivative of the
        parameter path lambda=phi(alpha) with respect to alpha, and w_i is the
        ensemble weight w(lambda).
        """
        # Execute block only if the integral weights are out-of-date.
        self.beginIfBlock(self.UPDATE_INTEGRAL_WEIGHTS_VARNAME + ' > 0')

        # Pre-compute all the potential associated to Gibbs, interpolated and
        # parametric force groups for all quadrature nodes. The nonparametric
        # part of the potential simplify in the integral so we can ignore it.
        self._add_compute_interpolated_nodes_energy()
        self._add_compute_parametric_nodes_energy()

        # Precompute all the variable names for readability.
        varnames = []
        for node_idx in range(self._n_quadrature_nodes):
            varnames.append([self.NODE_ENERGY_PREFIX + str(node_idx),
                             self.NODE_INTEGRAL_WEIGHT_PREFIX + str(node_idx),
                             self.NODE_EXPANDED_ENSEMBLE_WEIGHT_PREFIX + str(node_idx)])

        # Compute the minimum node energy. We'll subtract this from each exponent
        # to enhance the numerical stability of the sum of exp. This constant term
        # cancels out with the normalizing constant in the denominator.
        # max_expression = min(node_energy_1, min(node_energy_2, min(...)))
        assert self._n_quadrature_nodes > 1
        min_expression = 'min(' + varnames[0][0] + ','
        for node_idx in range(1, self._n_quadrature_nodes-1):
            min_expression +=  'min(' + varnames[node_idx][0] + ','
        min_expression += varnames[-1][0] + ')' * (self._n_quadrature_nodes-1)
        self.addComputeGlobal(self.MIN_NODE_ENERGY_VARNAME, min_expression)

        # Compute the integral weights of all quadrature nodes. Again, the
        # nonparametric terms of the potential cancel out.
        # integral_weight_expression = B_i * exp(-(energy_i - max_energy)/kT) * w_i
        integral_weight_expression = '{} * exp(-({} - ' + self.MIN_NODE_ENERGY_VARNAME + ')/kT) * {}'
        for node_idx, (energy_varname, integral_weight_varname, ee_weight_varname) in enumerate(varnames):
            node_expression = integral_weight_expression.format(
                self._quadrature_weights[node_idx], energy_varname, ee_weight_varname)
            self.addComputeGlobal(integral_weight_varname, node_expression)

        # Compute normalizing constant as the sum of the integral weights.
        integral_weight_varnames = [varname[1] for varname in varnames]
        integral_weight_sum_expression = ' + '.join(integral_weight_varnames)
        self.addComputeGlobal(self.INTEGRAL_WEIGHT_SUM_VARNAME, integral_weight_sum_expression)

        # Normalize the integral weights.
        for node_idx, varname in enumerate(integral_weight_varnames):
            self.addComputeGlobal(varname, varname + ' / ' + self.INTEGRAL_WEIGHT_SUM_VARNAME)

        # Update the cache flags.
        self.addComputeGlobal(self.UPDATE_INTEGRAL_WEIGHTS_VARNAME, '0.0')
        self.addComputeGlobal(self.UPDATE_AVERAGE_FORCE_VARNAME, '1.0')

        self.endBlock()

    def _add_compute_interpolated_nodes_energy(self):
        """Compute the contribution of Gibbs and interpolated forces to the potential at each quadrature node."""
        if len(self._gibbs_force_groups) == 0 and len(self._interpolated_force_groups) == 0:
            return

        # First cache all the force group energies that are required for the calculation.
        # -------------------------------------------------------------------------------
        # Compute the energies of the Gibbs forces.
        for force_group_idx, varname in self._get_force_group_energy_varnames(only_gibbs=True).items():
            self.addComputeGlobal(varname, 'energy' + str(force_group_idx))

        # Compute the energy for the interpolated forces at the initial and final
        # state. The energy will be linearly interpolated in between those two states.
        interpolated_energy_varnames = self._get_interpolated_force_group_energy_varnames()
        for force_group_idx, parameter_name in self._interpolated_force_groups:
            # Get the initial and final parameters.
            parameter_values = [self._quadrature_nodes_by_parameter[parameter_name][i] for i in [0, -1]]
            for varname, parameter_value in zip(interpolated_energy_varnames[force_group_idx], parameter_values):
                self.addComputeGlobal(parameter_name, str(parameter_value))
                self.addComputeGlobal(varname, 'energy' + str(force_group_idx))

        # Compute the contribution to the nodes' energies.
        # ------------------------------------------------
        for node_idx, node in enumerate(self._quadrature_nodes_by_node):
            node_energy_terms = []

            # Add terms for Gibbs forces.
            for force_group_idx, varname in self._get_force_group_energy_varnames(only_gibbs=True).items():
                lambda_value = str(node[force_group_idx])
                node_energy_terms.append('{} * {}'.format(lambda_value, varname))

            # Add terms for interpolated forces.
            for force_group_idx, parameter_name in self._interpolated_force_groups:
                interpolating_parameter = self._get_interpolating_parameter(parameter_name, node_idx)
                initial_energy_varname, final_energy_varname = interpolated_energy_varnames[force_group_idx]
                node_energy_terms.append('{0}*{1} + (1 - {0})*{2}'.format(
                    interpolating_parameter, initial_energy_varname, final_energy_varname))

            # Add compute node energy.
            node_energy_expression = ' + '.join(node_energy_terms)
            self.addComputeGlobal(self.NODE_ENERGY_PREFIX + str(node_idx), node_energy_expression)

    def _add_compute_parametric_nodes_energy(self):
        """Compute the contribution of parametric forces to the potential at each quadrature node."""
        if len(self._parametric_force_groups) == 0:
            return

        force_group_energy_varnames = self._get_force_group_energy_varnames(only_parametric=True)

        # The expression for the node energy is always the same.
        node_energy_terms = [force_group_energy_varnames[i] for i, _ in self._parametric_force_groups]
        node_energy_expression = ' + '.join(node_energy_terms)

        old_node = None
        for node_idx, node in enumerate(self._quadrature_nodes_by_node):

            # Recompute the energies of a force group only if the parameter value has changed.
            for force_group_idx, parameter_name in self._parametric_force_groups:
                if old_node is None or old_node[parameter_name] != node[parameter_name]:
                    self.addComputeGlobal(parameter_name, str(node[parameter_name]))
                    self.addComputeGlobal(force_group_energy_varnames[force_group_idx], 'energy'+str(force_group_idx))
            old_node = node

            # Update the node energy.
            node_energy_varname = self.NODE_ENERGY_PREFIX + str(node_idx)
            self.addComputeGlobal(node_energy_varname, node_energy_varname + ' + ' + node_energy_expression)

    def _add_compute_average_parameters(self):
        """Compute the average parameters for Gibbs and interpolated forces."""
        for force_group_idx in self._gibbs_force_groups:
            self._add_compute_force_group_average_parameter(force_group_idx, force_group_idx)
        for force_group_idx, parameter_name in self._interpolated_force_groups:
            self._add_compute_force_group_average_parameter(force_group_idx, parameter_name)

    def _add_compute_force_group_average_parameter(self, force_group_idx, parameter_name):
        """Compute the average parameter of a single force group."""
        # Check if this is an interpolated force or a Gibbs force. If this is a Gibbs
        # force, parameter_name is actually the index of the associated force group.
        is_interpolated = isinstance(parameter_name, str)

        # Build all the quadrature terms parameter_value_i * integral_weight_i.
        integral_terms = []
        for node_idx, parameter_value in enumerate(self._quadrature_nodes_by_parameter[parameter_name]):
            integral_weight_varname = self.NODE_INTEGRAL_WEIGHT_PREFIX + str(node_idx)
            if is_interpolated:
                parameter_value = self._get_interpolating_parameter(parameter_name, node_idx)
            integral_terms.append('{} * {}'.format(parameter_value, integral_weight_varname))

        # Add compute the average parameter.
        average_parameter_expression = '(' + ' + '.join(integral_terms) + ')'
        average_parameter_varname = self.AVERAGE_PARAMETER_PREFIX + str(force_group_idx)
        self.addComputeGlobal(average_parameter_varname, average_parameter_expression)

    def _add_update_online_estimates(self):
        """Update the estimate of the partition functions and the weights."""
        self._add_compute_integral_weights()

        self.addComputeGlobal(self.N_STEPS_VARNAME, self.N_STEPS_VARNAME + ' + 1')

        # General expression for partition function update.
        z_update_expression = '1/{n_steps} * exp(-({node_energy} - {min_energy})/kT)/{normalizing_constant} + '
        z_update_expression += '({n_steps}-1)/{n_steps} * {z}'
        # Update partition function and weight update at all quadrature nodes.
        for node_idx in range(self._n_quadrature_nodes):
            partition_function_varname = self.NODE_PARTITION_FUNCTION_PREFIX + str(node_idx)

            # The integral weight normalizing constant has been computed up to a factor
            # for stability which cancels out with the numerator. The nonparametric
            # components of the Hamiltonian also cancel out.
            z_expression = z_update_expression.format(
                n_steps=self.N_STEPS_VARNAME,
                node_energy=self.NODE_ENERGY_PREFIX + str(node_idx),
                min_energy=self.MIN_NODE_ENERGY_VARNAME,
                normalizing_constant=self.INTEGRAL_WEIGHT_SUM_VARNAME,
                z=partition_function_varname
            )
            self.addComputeGlobal(partition_function_varname, z_expression)

        # Update weights only if requested.
        self.beginIfBlock(self.EXPANDED_ENSEMBLE_WEIGHT_STEPSIZE_VARNAME + ' != 0.0')

        w_update_expression = '(1 - {gamma} * dt)*{w} + {gamma} * dt / {z}'
        for node_idx in range(self._n_quadrature_nodes):
            expanded_ensemble_weight_varname = self.NODE_EXPANDED_ENSEMBLE_WEIGHT_PREFIX + str(node_idx)
            w_expression = w_update_expression.format(
                gamma=self.EXPANDED_ENSEMBLE_WEIGHT_STEPSIZE_VARNAME,
                w=expanded_ensemble_weight_varname,
                z=self.NODE_PARTITION_FUNCTION_PREFIX + str(node_idx)
            )
            self.addComputeGlobal(expanded_ensemble_weight_varname, w_expression)

        # Normalize weights.
        w_normalizing_constant_terms = ['{} * {}{}'.format(quadrature_weight, self.NODE_EXPANDED_ENSEMBLE_WEIGHT_PREFIX, node_idx)
                                        for node_idx, quadrature_weight in enumerate(self._quadrature_weights)]
        w_normalizing_constant_expr = ' + '.join(w_normalizing_constant_terms)
        self.addComputeGlobal(self.EXPANDED_ENSEMBLE_WEIGHT_NORMALIZING_CONSTANT_VARNAME, w_normalizing_constant_expr)
        for node_idx in range(self._n_quadrature_nodes):
            expanded_ensemble_weight_varname = self.NODE_EXPANDED_ENSEMBLE_WEIGHT_PREFIX + str(node_idx)
            normalization_expression = '{}/{}'.format(expanded_ensemble_weight_varname,
                                                      self.EXPANDED_ENSEMBLE_WEIGHT_NORMALIZING_CONSTANT_VARNAME)
            self.addComputeGlobal(expanded_ensemble_weight_varname, normalization_expression)

        # The Boltzmann factors must be re-evaluated after changing the weights.
        self.addComputeGlobal(self.UPDATE_INTEGRAL_WEIGHTS_VARNAME, '1.0')
        self.addComputeGlobal(self.UPDATE_AVERAGE_FORCE_VARNAME, '1.0')

        self.endBlock()  # if(ee_weight_stepsize != 0.0)
