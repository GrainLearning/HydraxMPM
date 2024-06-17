# """Integration test for two sphere impact example."""

# import sys
# import unittest
# from os import listdir, remove,chdir
# from os.path import dirname, join, realpath


# class TestTwoSphereImpact(unittest.TestCase):
#     """Two sphere impact integration test."""

#     @staticmethod
#     def test_two_sphere_impact():
#         """Simulate two sphere impact."""
#         test_folder = dirname(realpath(__file__))

#         benchmark_folder = join(test_folder, "../examples/two_sphere_impact")

#         sys.path.insert(0, benchmark_folder)
#         chdir(test_folder)
#         import two_sphere_impact  # type: ignore # noqa

#         assert True, "Simulation failed"

#         print("Cleaning up output folder")
#         files = list(listdir(join(test_folder, "output")))
#         for f in files:
#             output_file = join(test_folder, "output", f)
#             remove(output_file)
#             print(f"Removed {output_file}")

#     @staticmethod
#     def test_cube_fall():
#         """Simulate cube fall."""
#         test_folder = dirname(realpath(__file__))

#         benchmark_folder = join(test_folder, "../examples/cube_fall")

#         sys.path.insert(0, benchmark_folder)
#         chdir(test_folder)
#         import cube_fall_rough_walls

#         print("Cleaning up output folder")
#         files = list(listdir(join(test_folder, "output")))
#         for f in files:
#             output_file = join(test_folder, "output", f)
#             remove(output_file)
#             print(f"Removed {output_file}")


# if __name__ == "__main__":
#     unittest.main()
