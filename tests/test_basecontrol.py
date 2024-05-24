"""Unit tests for the base single element benchmark."""

import unittest

import numpy as np

import pymudokon as pm


class TestBaseControl(unittest.TestCase):
    """Base Control."""

    @staticmethod
    def test_init():
        """Unit test to test initialization."""
        material = pm.LinearIsotropicElastic.register(E=1000.0, nu=0.2, num_particles=2, dim=3)

        basecontrol = pm.BaseControl.register(
            material=material,
            results_to_store=["stress", "strain", "eps_e"],
        )

        expected_results = dict({"stress": [], "strain": [], "eps_e": []})
        print(basecontrol.results)
        np.testing.assert_equal(basecontrol.results, expected_results)


if __name__ == "__main__":
    unittest.main()
