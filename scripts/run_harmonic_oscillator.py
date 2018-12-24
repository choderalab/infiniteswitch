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


# ==============================================================================
# Harmonic oscillator simulations.
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


class HarmonicOscillatorSimulation:
    """Infinite switch serialize tempering with harmonic oscillator."""

    REFERENCE_BETA = 1/(kB * 250.0*unit.kelvin)
    END_POINTS_BETA = [1/(kB * 200.0*unit.kelvin), 1/(kB * 300.0*unit.kelvin)]

    def __init__(self, timestep=2.0*unit.femtoseconds, n_quadrature_nodes=20):
        """Shared test cases for the test suite."""
        self.n_quadrature_nodes = n_quadrature_nodes

        # Construct the temperature path.
        beta_a, beta_b = [self.END_POINTS_BETA[i]/self.REFERENCE_BETA for i in [0, 1]]
        temperature_ladder, temperature_ladder_derivative = get_linearly_interpolated_path(beta_a, beta_b)

        # Configure quadrature nodes.
        self.quadrature_nodes, self.quadrature_weights, self.path_derivative = determine_quadrature(
            temperature_ladder, temperature_ladder_derivative, deg=self.n_quadrature_nodes)
        self.end_point_nodes = [temperature_ladder(node) for node in [-1, 1]]

        # Create and configure simulation object.
        harmonic_oscillator = mmtools.testsystems.HarmonicOscillator(mass=12.0*unit.amu)
        system = harmonic_oscillator.system
        self.K = harmonic_oscillator.K

        # The protocol modifies the potential of the only force in the system.
        assert system.getNumForces() == 1
        self.quadrature_nodes = {0: [node[0] for node in self.quadrature_nodes]}

        # Create infinite switch integrator.
        self.integrator = InfiniteSwitchIntegrator(system, self.quadrature_nodes, self.path_derivative,
                                                  self.quadrature_weights, timestep=timestep)
        self.kT = (kB * self.integrator.getTemperature()).value_in_unit_system(unit.md_unit_system)
        # Create context.
        self.context = openmm.Context(harmonic_oscillator.system, self.integrator)
        self.context.setPositions(harmonic_oscillator.positions)
        self.context.setVelocitiesToTemperature(self.integrator.getTemperature())

        # Compute expected free energy differences.
        f_i_analytical = []
        # TODO for beta_i in [self.end_point_nodes[0]] + self.quadrature_nodes + [self.end_point_nodes[-1]]:
        for beta_i in self.quadrature_nodes[0]:
            temperature = 1 / (beta_i * self.REFERENCE_BETA) / kB
            log_z = compute_harmonic_oscillator_log_z(harmonic_oscillator.K, temperature)
            f_i_analytical.append(log_z)
        self.Df_ij_analytical = compute_Df_ij(f_i_analytical)

        # Step on integrator to generate variables.
        self.integrator.step(1)

    def run_harmonic_oscillator(self, n_iterations, n_steps_per_iteration):
        """Run the system and collect the free energy trajectories."""
        for trajectory_name in ['z_i_computed', 'Df_i_computed', 'ee_weights']:
            setattr(self, trajectory_name, np.empty((self.n_quadrature_nodes, n_iterations)))
        self.positions = np.empty((3, n_iterations))

        for iteration in range(n_iterations):
            self.integrator.step(n_steps_per_iteration)
            z_i_computed = np.array(self.integrator.get_nodes_partition_functions())
            self.z_i_computed[:, iteration] = z_i_computed
            self.Df_i_computed[:, iteration] = compute_Df_ij(-np.log(z_i_computed))[0]
            self.ee_weights[:, iteration] = self.integrator.get_nodes_expanded_ensemble_weights()
            self.positions[:, iteration] = self.context.getState(getPositions=True).getPositions(asNumpy=True)[0]

    def save_data(self, data_directory):
        os.makedirs(data_directory, exist_ok=True)
        for attribute_name in ['z_i_computed', 'Df_i_computed', 'ee_weights', 'positions']:
            np.save(os.path.join(data_directory, attribute_name + '.npy'), getattr(self, attribute_name))

    def restore_data(self, data_directory):
        for attribute_name in ['z_i_computed', 'Df_i_computed', 'ee_weights', 'positions']:
            setattr(self, attribute_name, np.load(os.path.join(data_directory, attribute_name + '.npy')))

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
        palette = sns.color_palette('coolwarm', n_colors=simulation.n_quadrature_nodes)

        # Free energy trajectories.
        fig1, ax1 = plt.subplots()
        for i, Df_trajectory in enumerate(self.Df_i_computed):
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
        for i, w_trajectory in enumerate(self.ee_weights):
            ax2.plot(w_trajectory, color=palette[i])
        ax2.set_xlabel('iteration')
        ax2.set_ylabel('expanded ensemble weight')
        save(fig2, 'ee_weights.pdf')

        # X-coordinate.
        fig3, ax3 = plt.subplots()
        radius = [np.linalg.norm(pos) for pos in self.positions.transpose()]
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

    simulation = HarmonicOscillatorSimulation(n_quadrature_nodes=50)
    simulation.run_harmonic_oscillator(n_iterations=2000, n_steps_per_iteration=500)
    simulation.save_data(harmonic_oscillator_dir)
    # simulation.restore_data(harmonic_oscillator_dir)
    # simulation.analyze(harmonic_oscillator_dir)


