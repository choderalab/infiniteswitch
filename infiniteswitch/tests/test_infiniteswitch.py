#!/usr/bin/env python

"""
Unit and regression test for the infiniteswitch module.
"""

# ==============================================================================
# Global imports.
# ==============================================================================

import collections
import copy

import numpy as np
import openmmtools as mmtools
from openmmtools.constants import kB
from simtk import openmm, unit

from nose.tools import assert_raises_regexp

from infiniteswitch import (InfiniteSwitchIntegrator,
                            get_linearly_interpolated_path,
                            determine_quadrature)


# ==============================================================================
# Reference implementations for tests.
# ==============================================================================

def compute_nodes_energies(context, integrator):
    """Compute the terms of the potential that depend on the Hamiltonian parameters.

    Returns
    -------
    nodes_energies : List[float]
        The list of the potential energies (in kJ/mol) evaluated at each node,
        ignoring the contribution from the nonparametric forces.
    """
    energy_unit = unit.kilojoule_per_mole

    # Start by getting the energy of the Gibbs force groups.
    gibbs_energies = {}
    for force_group in integrator._gibbs_force_groups:
        potential = context.getState(getEnergy=True, groups={force_group}).getPotentialEnergy()
        gibbs_energies[force_group] = potential / energy_unit

    # Get the initial and final interpolated energies.
    initial_node, final_node = [integrator._quadrature_nodes_by_node[i] for i in [0, -1]]
    interpolated_energies = {force_group: [] for force_group, _ in integrator._interpolated_force_groups}
    for node in [initial_node, final_node]:
        for force_group, parameter_name in integrator._interpolated_force_groups:
            context.setParameter(parameter_name, node[parameter_name])
            potential = context.getState(getEnergy=True, groups={force_group}).getPotentialEnergy()
            interpolated_energies[force_group].append(potential / energy_unit)

    # Compute energy for each node.
    nodes_energies = []
    for node in integrator._quadrature_nodes_by_node:
        potential = 0

        # Compute contribution from Gibbs forces.
        for force_group in integrator._gibbs_force_groups:
            potential += node[force_group] * gibbs_energies[force_group]

        # Compute contribution from interpolated forces.
        for force_group, parameter_name in integrator._interpolated_force_groups:
            parameter_value = node[parameter_name]
            initial_value, final_value = initial_node[parameter_name], final_node[parameter_name]
            initial_energy, final_energy = interpolated_energies[force_group]
            alpha = (parameter_value**2 - final_value**2) / (initial_value**2 - final_value**2)
            potential += alpha * initial_energy + (1 - alpha) * final_energy

        # Compute contribution for parametric forces.
        for force_group, parameter_name in integrator._parametric_force_groups:
            context.setParameter(parameter_name, node[parameter_name])
            state = context.getState(getEnergy=True, groups={force_group})
            potential += state.getPotentialEnergy() / energy_unit

        nodes_energies.append(potential)
    return nodes_energies


def compute_nodes_integral_weights(context, integrator, normalize=True, ee_weights=None):
    """Compute the integral weights of all nodes."""
    kT = integrator.getGlobalVariableByName('kT')
    node_energies = compute_nodes_energies(context, integrator)
    if ee_weights is None:
        ee_weights = integrator.get_nodes_expanded_ensemble_weights()

    # Compute the unnormalized integral weights naively.
    node_integral_weights = []
    for quadrature_weight, node_energy, ee_weight in zip(integrator._quadrature_weights, node_energies, ee_weights):
        integral_weight = quadrature_weight * np.exp(-node_energy/kT) * ee_weight
        node_integral_weights.append(integral_weight)

    # Normalize the integral weights.
    if normalize:
        normalizing_constant = sum(node_integral_weights)
        node_integral_weights = [w / normalizing_constant for w in node_integral_weights]
    return node_integral_weights


def compute_average_force(context, integrator):
    """Compute the force of the marginal distribution."""
    integral_weights = compute_nodes_integral_weights(context, integrator)
    average_force = np.zeros((context.getSystem().getNumParticles(), 3))

    def get_force(context, force_group_idx):
        force = context.getState(getForces=True, groups={force_group_idx}).getForces()
        return np.array(force.value_in_unit_system(unit.md_unit_system))

    # Compute contribution Gibbs forces.
    for force_group_idx in integrator._gibbs_force_groups:
        parameter_values = integrator._quadrature_nodes_by_parameter[force_group_idx]
        average_lambda = sum(l * w for l, w in zip(parameter_values, integral_weights))
        average_force += average_lambda * get_force(context, force_group_idx)

    # Compute contribution of interpolated forces.
    for force_group_idx, parameter_name in integrator._interpolated_force_groups:
        initial_value, final_value = [integrator._quadrature_nodes_by_parameter[parameter_name][i] for i in [0, -1]]
        average_alpha = 0.0
        for parameter_value, w in zip(integrator._quadrature_nodes_by_parameter[parameter_name], integral_weights):
            average_alpha += (parameter_value**2 - final_value**2) / (initial_value**2 - final_value**2) * w
        context.setParameter(parameter_name, initial_value)
        average_force += average_alpha * get_force(context, force_group_idx)
        context.setParameter(parameter_name, final_value)
        average_force += (1 - average_alpha) * get_force(context, force_group_idx)

    # Compute contribution of parametric forces.
    for force_group_idx, parameter_name in integrator._parametric_force_groups:
        for parameter_value, w in zip(integrator._quadrature_nodes_by_parameter[parameter_name], integral_weights):
            context.setParameter(parameter_name, parameter_value)
            average_force += w * get_force(context, force_group_idx)

    # Contribution of nonparametric forces.
    for force_group_idx in integrator._nonparametric_force_groups:
        average_force += get_force(context, force_group_idx)

    return average_force


def compute_online_estimates(context, integrator):
    """Reference implementation of the partition function and expanded ensemble weight updates."""
    # Test pre-condition: This assume that only a single update has been done.
    n_steps = integrator.getGlobalVariableByName(integrator.N_STEPS_VARNAME)
    assert n_steps == 1

    node_energies = compute_nodes_energies(context, integrator)
    n_quadrature_nodes = len(node_energies)

    # Pass the original EE weights to compute the integral weights.
    initial_w = 1 / sum(integrator._quadrature_weights)
    ee_weights = [initial_w for _ in range(n_quadrature_nodes)]
    integral_weights = compute_nodes_integral_weights(context, integrator, normalize=False, ee_weights=ee_weights)
    normalizing_constant = sum(integral_weights)

    dt = integrator.getStepSize().value_in_unit_system(unit.md_unit_system)
    gamma = integrator.ee_weight_stepsize
    kT = integrator.kT.value_in_unit_system(unit.md_unit_system)

    partition_functions = []
    expanded_ensemble_weights = []
    for node_idx, energy in enumerate(node_energies):
        z = 1/n_steps * np.exp(-energy/kT) / normalizing_constant  # + (n_steps-1)/n_steps * z
        w = (1 - gamma * dt) * initial_w + gamma * dt / z
        partition_functions.append(z)
        expanded_ensemble_weights.append(w)

    normalizing_constant = sum(b*w for b, w in zip(integrator._quadrature_weights, expanded_ensemble_weights))
    expanded_ensemble_weights = [expanded_ensemble_weights[i]/normalizing_constant
                                 for i in range(n_quadrature_nodes)]

    return partition_functions, expanded_ensemble_weights


# ==============================================================================
# Tests.
# ==============================================================================

def test_get_linearly_interpolated_path():
    """Test the construction of linear paths."""
    # Each test case is an (initial, final) pair.
    test_cases =  [
        ([0.0, -1.0, 0.0], [2.0, 3.0, 0.0]),
        (200, 400)
    ]
    for initial, final in test_cases:
        path, derivative = get_linearly_interpolated_path(initial, final)
        assert np.allclose(path(-1), initial)
        assert np.allclose(path(1), final)


def test_multiple_hamiltonian_parameters():
    """Test that InfiniteSwitchIntegrator raises an error with unsopported forces."""
    single_parameter_force = openmm.CustomBondForce('0.0;')
    single_parameter_force.addGlobalParameter('parameter1', 1.0)
    two_parameter_force =  openmm.CustomBondForce('0.0;')
    two_parameter_force.addGlobalParameter('parameter2', 1.0)
    two_parameter_force.addGlobalParameter('parameter3', 1.0)

    system = openmm.System()
    system.addForce(single_parameter_force)
    system.addForce(two_parameter_force)

    # Use a proxy object to isolate the testing to the private method.
    class proxy:
        _interpolated_force_groups = []
        _gibbs_force_groups = []
        _parametric_force_groups = []
        _nonparametric_force_groups = []

    def generator(nodes, regex):
        with assert_raises_regexp(ValueError, regex):
            InfiniteSwitchIntegrator._optimize_force_groups(proxy, system, nodes)

    test_cases = [
        ({0: [1.0], 'parameter1': [0.5]}, 'both in Gibbs ensemble and controlled by the Hamiltonian parameters'),
        ({'parameter2': [1.0], 'parameter3': [0.5]}, 'controlled by multiple Hamiltonian parameters'),
        ({'parameter4': [1.0]}, 'Cannot find some of the Hamiltonian parameters')
    ]
    for quadrature_nodes, regex_error in test_cases:
        yield generator, quadrature_nodes, regex_error


def test_optimize_force_groups():
    """Test that force are grouped correctly into nonparametric, parametric, Gibbs, and nonbonded force groups."""
    nonbonded_force1 = openmm.NonbondedForce()
    nonbonded_force2 = openmm.NonbondedForce()
    nonbonded_force2.addGlobalParameter('offset_parameter', 1.0)
    nonbonded_force3 = openmm.NonbondedForce()
    nonbonded_force3.addGlobalParameter('parameter1', 1.0)

    parametric_force1 = openmm.CustomNonbondedForce('0.0;')
    parametric_force1.addGlobalParameter('parameter1', 1.0)
    parametric_force2 = openmm.CustomBondForce('0.0;')
    parametric_force2.addGlobalParameter('parameter1', 1.0)
    parametric_force3 = openmm.CustomTorsionForce('0.0;')
    parametric_force3.addGlobalParameter('parameter2', 1.0)

    nonparametric_force1 = openmm.HarmonicBondForce()
    nonparametric_force2 = openmm.HarmonicAngleForce()

    forces = [nonbonded_force1, nonbonded_force2, nonbonded_force3, parametric_force1,
              parametric_force2, parametric_force3, nonparametric_force1, nonparametric_force2]
    system = openmm.System()
    for force in forces:
        system.addForce(force)

    # Use a proxy object to isolate the testing to the private method.
    class Proxy:
        def __init__(self):
            self._interpolated_force_groups = []
            self._gibbs_force_groups = []
            self._parametric_force_groups = []
            self._nonparametric_force_groups = []

    def generator(system, quadrature_nodes, expected_groups, gibbs_force_groups,
                  interpolated_force_groups, parametric_force_groups, nonparametric_force_groups):
        proxy = Proxy()
        InfiniteSwitchIntegrator._optimize_force_groups(proxy, system, quadrature_nodes)
        assert proxy._gibbs_force_groups == gibbs_force_groups
        assert proxy._interpolated_force_groups == interpolated_force_groups
        assert proxy._parametric_force_groups == parametric_force_groups
        assert proxy._nonparametric_force_groups == nonparametric_force_groups
        for force_idx, expected_group in enumerate(expected_groups):
            assert system.getForce(force_idx).getForceGroup() == expected_group

    Case = collections.namedtuple('Case', 'quadrature_nodes expected_groups gibbs_force_groups '
                                          'interpolated_force_groups parametric_force_groups '
                                          'nonparametric_force_groups')
    test_cases = [
        # NonbondedForces with offset parameters get split into their own force group.
        Case(
            quadrature_nodes={'parameter1': [1.0], 'parameter2': [1.0]},
            expected_groups=[3, 3, 0, 1, 1, 2, 3, 3],
            gibbs_force_groups=[],
            interpolated_force_groups=[(0, 'parameter1')],
            parametric_force_groups=[(1, 'parameter1'), (2, 'parameter2')],
            nonparametric_force_groups=[3]
        ),
        # Gibbs forces with the same protocol get grouped in the same force group.
        Case(
            quadrature_nodes={0: [1.0], 6: [1.0], 7: [0.5], 'offset_parameter': [1.0], 'parameter1': [1.0]},
            expected_groups=[0, 2, 3, 4, 4, 5, 0, 1],
            gibbs_force_groups=[0, 1],
            interpolated_force_groups=[(2, 'offset_parameter'), (3, 'parameter1')],
            parametric_force_groups=[(4, 'parameter1')],
            nonparametric_force_groups=[5]
        )
    ]
    for test_case in test_cases:
        yield (generator, copy.deepcopy(system), *test_case)


class TestInfiniteSwitchIntegrator:

    @classmethod
    def setup_class(cls):
        """Set up system and parameters shared by tests."""
        # Create an alchemical system with electrostatics softened through the
        # NondondedForce offset parameters so that we can test an interpolated force.
        alanine = mmtools.testsystems.AlanineDipeptideVacuum()
        system = alanine.system
        _, nonbonded_force = mmtools.forces.find_forces(system, openmm.NonbondedForce, only_one=True)
        nonbonded_force.setNonbondedMethod(openmm.NonbondedForce.PME)

        from openmmtools.alchemy import AlchemicalRegion, AbsoluteAlchemicalFactory
        region = AlchemicalRegion(alchemical_atoms=range(15), annihilate_sterics=False)
        factory = AbsoluteAlchemicalFactory(alchemical_pme_treatment='exact')
        cls.system = factory.create_alchemical_system(system, region)
        cls.positions = alanine.positions

        # This protocol has the torsion force in Gibbs ensemble, electrostatics
        # is interpolated, and lambda_sterics is treated with the general equation.
        torsion_force_index, torsion_force = mmtools.forces.find_forces(system, openmm.PeriodicTorsionForce, only_one=True)
        cls.quadrature_nodes = {
            torsion_force_index: [1.0, 0.5, 0.25],
            'lambda_electrostatics': [1.0, 0.0, 0.0],
            'lambda_sterics': [1.0, 1.0, 0.0]
        }
        cls.quadrature_nodes_gradient_norm = [1.0, 1.0, 1.0]
        cls.quadrature_weights = [0.25, 0.5, 0.25]

    @classmethod
    def create_context(cls, **kwargs):
        """Create a new context with an infinite switch integrator."""
        integrator = InfiniteSwitchIntegrator(cls.system, cls.quadrature_nodes,
                                              cls.quadrature_nodes_gradient_norm,
                                              cls.quadrature_weights, **kwargs)
        # Initialize the Context and step to allow the computation of the energies.
        context = openmm.Context(cls.system, integrator)
        context.setPositions(cls.positions)
        integrator.step(1)
        return context

    def test_compute_node_energies(self):
        """Test that energies computed for each quadrature node are correct."""
        context = self.create_context()
        integrator = context.getIntegrator()

        # Compare the integrator energies with the reference implementation.
        integrator_energies = integrator.get_nodes_energies()
        reference_energies = compute_nodes_energies(context, integrator)
        assert np.allclose(integrator_energies, reference_energies), (integrator_energies, reference_energies)

    def test_compute_integral_weights(self):
        """Test that the integral weights computed for each quadrature node are correct."""
        # Deactivate the online estimate, after which the integral
        # weights stored in the integrator are out-of-date.
        context = self.create_context(ee_weight_stepsize=0.0)
        integrator = context.getIntegrator()

        # Test pre-condition.
        assert integrator.ee_weight_stepsize == 0.0, integrator.ee_weight_stepsize

        # Compare the integrator integral weights with the reference implementation.
        integrator_integral_weights = integrator.get_nodes_integral_weights()
        reference_integral_weights = compute_nodes_integral_weights(context, integrator)
        assert np.allclose(integrator_integral_weights, reference_integral_weights), (integrator_integral_weights, reference_integral_weights)

    def test_compute_average_force(self):
        """Test that the calculation of the marginal distribution force is correct."""
        # Deactivate the online estimate, after which the integral
        # weights stored in the integrator are out-of-date.
        context = self.create_context(ee_weight_stepsize=0.0)
        integrator = context.getIntegrator()

        # Compare the integrator forces with the reference implementation.
        integrator_average_force = np.array(integrator.get_forces())
        reference_average_force = compute_average_force(context, integrator)
        assert np.allclose(integrator_average_force, reference_average_force), (integrator_average_force, reference_average_force)

    def test_update_online_estimates(self):
        """Test that the online estimates of the partition function and weights are correct."""
        context = self.create_context()
        integrator = context.getIntegrator()

        # Compare the integrator partition functions and expanded ensemble weight with the reference implementation.
        integrator_partition_functions = integrator.get_nodes_partition_functions()
        integrator_ee_weights = integrator.get_nodes_expanded_ensemble_weights()
        reference_partition_functions, reference_ee_weights = compute_online_estimates(context, integrator)
        assert np.allclose(integrator_partition_functions, reference_partition_functions), (integrator_partition_functions, reference_partition_functions)
        assert np.allclose(integrator_ee_weights, reference_ee_weights), (integrator_ee_weights, reference_ee_weights)

        # Check that the expanded ensemble weights are appropriately normalized.
        ee_weights_integral = sum(b*w for b, w in zip(integrator._quadrature_weights, integrator_ee_weights))
        assert np.isclose(ee_weights_integral, 1.0), ee_weights_integral
