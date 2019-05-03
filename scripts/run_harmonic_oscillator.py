#!/usr/bin/env python

# ==============================================================================
# Global imports.
# ==============================================================================

import os

import numpy as np
import openmmtools as mmtools
from openmmtools.constants import kB
from simtk import openmm, unit

from infiniteswitch import (InfiniteSwitchIntegrator,
                            get_linearly_interpolated_path,
                            determine_quadrature)
from infiniteswitch.storage import Storage, NetCDFArray


# ==============================================================================
# Helper functions to analyze the harmonic oscillator.
# ==============================================================================

def compute_harmonic_oscillator_log_z(K, temperature):
    """Compute a 3D harmonic oscillator.

    Parameters
    ----------
    K : simtk.unit.Quantity
        Spring constant.
    temperature : simtk.unit.Quantity
        Temperature.

    Returns
    -------
    f : float
        The unit-less free energy.

    """
    # Compute thermal energy and inverse temperature from specified temperature.
    beta = 1.0 / (kB * temperature)  # inverse temperature
    # Compute standard deviation along one dimension.
    sigma = 1.0 / unit.sqrt(beta * K)
    # Compute dimensionless free energy.
    return - np.log(2 * np.pi * (sigma / unit.angstroms)**2) * (3.0/2.0)


def compute_Df_ij(f_i):
    # Compute analytical Delta_f_ij
    n_states = len(f_i)
    Df_ij = np.zeros([n_states, n_states], np.float64)
    for i in range(n_states):
        for j in range(n_states):
            Df_ij[i, j] = f_i[j] - f_i[i]
    return Df_ij


# ==============================================================================
# Harmonic oscillator simulations.
# ==============================================================================

def create_storage_schema(n_quadrature_nodes):
        """Create the Storage schema with the appropriate data dimensions."""

        class SimulationStorage(Storage):
            z_i_computed = NetCDFArray(
                relative_file_path='storage.nc',
                variable_path='z_i_computed',
                datatype='f8',
                is_appendable=True,
                dimensions=[('iteration1', 0), ('quadrature_node', n_quadrature_nodes)]
            )
            Df_i_computed = NetCDFArray(
                relative_file_path='storage.nc',
                variable_path='Df_i_computed',
                datatype='f8',
                is_appendable=True,
                dimensions=[('iteration2', 0), ('quadrature_node', n_quadrature_nodes)]
            )
            ee_weights = NetCDFArray(
                relative_file_path='storage.nc',
                variable_path='ee_weights',
                datatype='f8',
                is_appendable=True,
                dimensions=[('iteration3', 0), ('quadrature_node', n_quadrature_nodes)]
            )
            positions = NetCDFArray(
                relative_file_path='storage.nc',
                variable_path='positions',
                datatype='f8',
                is_appendable=True,
                dimensions=[('iteration4', 0), ('spatial', 3)]
            )

        return SimulationStorage


class HarmonicOscillatorSimulation:
    """Infinite switch serialize tempering with harmonic oscillator."""

    REFERENCE_BETA = 1/(kB * 250.0*unit.kelvin)
    END_POINTS_BETA = [1/(kB * 200.0*unit.kelvin), 1/(kB * 300.0*unit.kelvin)]

    def __init__(self, storage_directory, timestep=2.0*unit.femtoseconds, n_quadrature_nodes=20):
        self.n_quadrature_nodes = n_quadrature_nodes

        # Create/restore storage.
        Schema = create_storage_schema(n_quadrature_nodes)
        self.storage = Schema(storage_directory, open_mode='a')

        # Create and configure simulation object.
        harmonic_oscillator = mmtools.testsystems.HarmonicOscillator(mass=12.0*unit.amu)
        system = harmonic_oscillator.system
        self.K = harmonic_oscillator.K

        # Construct the temperature path.
        beta_a, beta_b = [self.END_POINTS_BETA[i]/self.REFERENCE_BETA for i in [0, 1]]
        temperature_ladder, temperature_ladder_derivative = get_linearly_interpolated_path(beta_a, beta_b)
        # Configure quadrature nodes.
        self.quadrature_nodes, self.quadrature_weights, self.path_derivative = determine_quadrature(
            temperature_ladder, temperature_ladder_derivative, deg=self.n_quadrature_nodes)
        # TODO: self.end_point_nodes = [temperature_ladder(node) for node in [-1, 1]]

        # The protocol modifies the potential of the only force in the system.
        assert system.getNumForces() == 1
        self.quadrature_nodes = {0: [node[0] for node in self.quadrature_nodes]}

        # Create infinite switch integrator.
        integrator = InfiniteSwitchIntegrator(system, self.quadrature_nodes, self.path_derivative,
                                              self.quadrature_weights, timestep=timestep)

        # Create context.
        self.context = openmm.Context(harmonic_oscillator.system, integrator)
        self.context.setPositions(harmonic_oscillator.positions)
        self.context.setVelocitiesToTemperature(integrator.getTemperature())

        # Compute expected free energy differences.
        f_i_analytical = []
        # TODO for beta_i in [self.end_point_nodes[0]] + self.quadrature_nodes + [self.end_point_nodes[-1]]:
        for beta_i in self.quadrature_nodes[0]:
            temperature = 1 / (beta_i * self.REFERENCE_BETA) / kB
            log_z = compute_harmonic_oscillator_log_z(harmonic_oscillator.K, temperature)
            f_i_analytical.append(log_z)
        self.Df_ij_analytical = compute_Df_ij(f_i_analytical)

    @property
    def integrator(self):
        return self.context.getIntegrator()

    def run_harmonic_oscillator(self, n_iterations, n_steps_per_iteration):
        """Run the system and collect the free energy trajectories."""
        for iteration in range(n_iterations):
            self.integrator.step(n_steps_per_iteration)
            self._store_iteration()

    def _store_iteration(self):
        z_i_computed = np.array(self.integrator.get_nodes_partition_functions())
        self.storage.z_i_computed.append(z_i_computed)
        computed_Df_ij = compute_Df_ij(-np.log(z_i_computed))[0]
        self.storage.Df_i_computed.append(computed_Df_ij)
        ee_weights = self.integrator.get_nodes_expanded_ensemble_weights()
        self.storage.ee_weights.append(ee_weights)
        positions = self.context.getState(getPositions=True).getPositions(asNumpy=True)[0]
        self.storage.positions.append(positions)

    def analyze(self, analysis_directory):
        """Plot the free energy trajectories for all quadrature nodes."""
        from matplotlib import pyplot as plt
        import seaborn as sns
        import scipy.integrate

        def save(fig, file_name):
            if analysis_directory is None:
                fig.show()
            else:
                file_path = os.path.join(analysis_directory, file_name)
                fig.savefig(file_path)

        sns.set_style('whitegrid')
        palette = sns.color_palette('coolwarm', n_colors=self.n_quadrature_nodes)

        # Free energy trajectories.
        fig1, ax1 = plt.subplots()
        for i, Df_trajectory in enumerate(self.storage.Df_i_computed.transpose()):
            ax1.plot(Df_trajectory, color=palette[i])
            # Plot reference line.
            reference_Df = self.Df_ij_analytical[0][i]
            ax1.plot([reference_Df for _ in range(len(Df_trajectory))], color=palette[i], label=str(i))
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('Df [kT]')
        ax1.legend()
        save(fig1, 'free_energies.pdf')

        # Expanded ensemble weights.
        fig2, ax2 = plt.subplots()
        for i, w_trajectory in enumerate(self.storage.ee_weights.transpose()):
            ax2.plot(w_trajectory, color=palette[i])
        ax2.set_xlabel('iteration')
        ax2.set_ylabel('expanded ensemble weight')
        save(fig2, 'ee_weights.pdf')

        # X-coordinate.
        fig3, ax3 = plt.subplots()
        radius = [np.linalg.norm(pos) for pos in self.storage.positions]
        # phi = 1/np.cos(self.positions[2]/radius)
        sns.distplot(radius, ax=ax3, label='simulated')
        histogram_height = max([h.get_height() for h in ax3.patches])
        mean_radius = np.mean(radius)
        ax3.plot([mean_radius for _ in range(100)], np.linspace(0, histogram_height, num=100))

        # Plot expected distribution.
        k = self.K.value_in_unit_system(unit.md_unit_system)
        beta_i, beta_f = [self.END_POINTS_BETA[i].value_in_unit_system(unit.md_unit_system) for i in [0, 1]]

        def conditional_p(r, beta):
            # Marginal probability over phi and theta given the radius r in polar coordinates.
            return (k*beta/2/np.pi)**(3/2) * np.exp(-beta * k/2 * r**2) * r**2 * 4*np.pi

        def marginal_p(r):
            # beta_i > beta_f so we reverse the integration sign.
            return -scipy.integrate.quad(lambda x: conditional_p(r, x), beta_i, beta_f)[0]

        radius_range = np.linspace(min(radius), max(radius), num=200)
        initial_distribution = np.array([conditional_p(r, beta_i) for r in radius_range])
        final_distribution = np.array([conditional_p(r, beta_f) for r in radius_range])
        marginal_distribution = np.array([marginal_p(r) for r in radius_range])
        # Scale the expected distribution to match the histogram.
        ax3.plot(radius_range, initial_distribution * histogram_height/max(initial_distribution), label='initial')
        ax3.plot(radius_range, final_distribution * histogram_height/max(final_distribution), label='final')
        ax3.plot(radius_range, marginal_distribution * histogram_height/max(marginal_distribution), label='marginal')
        ax3.legend()
        save(fig3, 'radius.pdf')


# ==============================================================================
# Main.
# ==============================================================================

if __name__ == '__main__':
    harmonic_oscillator_dir = os.path.join('..', 'data', 'harmonic_oscillator')
    simulation = HarmonicOscillatorSimulation(storage_directory=harmonic_oscillator_dir,
                                              n_quadrature_nodes=20)
    simulation.run_harmonic_oscillator(n_iterations=2000, n_steps_per_iteration=500)
    simulation.analyze(harmonic_oscillator_dir)


