import unittest
import os
import numpy as np
from src import kmeans_clustering as km  # Import your main K-Means script


class TestKMeansClustering(unittest.TestCase):
    """
    Unit tests for the K-Means clustering application.

    This class tests each key function in kmeans_clustering.py to ensure:
    1. Data loads correctly
    2. Preprocessing works and removes invalid values
    3. Scaling is correct
    4. K-Means clustering produces valid results
    5. Visualization runs without errors
    """

    @classmethod
    def setUpClass(cls):
        """
        This method runs once before all tests.
        It loads a small subset of the dataset to speed up testing.
        """
        # Use the data path defined in the main script
        cls.data_path = km.DATA_PATH

        # Load 1 CSV file only for testing (faster than full dataset)
        cls.df = km.load_data(cls.data_path, max_files=1)

        # Preprocess the data (convert to numeric, handle NaN/inf, clip extremes)
        cls.numeric_df = km.preprocess_data(cls.df)

        # Scale the numeric data for clustering
        cls.scaled_data = km.scale_data(cls.numeric_df)

    # -----------------------------
    # Test 1: Data Loading
    # -----------------------------
    def test_load_data_nonempty(self):
        """
        Ensure that the loaded DataFrame is not empty.
        """
        self.assertFalse(self.df.empty, "Loaded DataFrame should not be empty.")

    def test_load_data_columns(self):
        """
        Ensure that the loaded DataFrame has at least one column.
        """
        self.assertGreater(self.df.shape[1], 0, "DataFrame should have columns.")

    # -----------------------------
    # Test 2: Preprocessing
    # -----------------------------
    def test_preprocess_numeric(self):
        """
        Check that all preprocessed columns are numeric.
        """
        self.assertTrue(np.issubdtype(self.numeric_df.dtypes[0], np.number),
                        "Preprocessed columns should be numeric.")

    def test_preprocess_no_inf(self):
        """
        Confirm there are no infinity values in the preprocessed data.
        """
        self.assertFalse(np.isinf(self.numeric_df.values).any(),
                         "Preprocessed DataFrame should not contain inf values.")

    def test_preprocess_no_nan(self):
        """
        Confirm there are no NaN (missing) values in the preprocessed data.
        """
        self.assertFalse(self.numeric_df.isnull().values.any(),
                         "Preprocessed DataFrame should not contain NaN values.")

    # -----------------------------
    # Test 3: Scaling
    # -----------------------------
    def test_scaling_shape(self):
        """
        Check that the scaled data has the same shape (rows & columns) as the numeric data.
        """
        self.assertEqual(self.scaled_data.shape, self.numeric_df.shape,
                         "Scaled data should have the same shape as numeric DataFrame.")

    def test_scaling_values(self):
        """
        Check that scaled data has standard deviation ~1 for columns with actual variance.
        Columns that were constant in the original data are ignored.
        """
        stds = np.std(self.scaled_data, axis=0)  # Standard deviation of scaled data
        original_stds = np.std(self.numeric_df.values, axis=0)  # Original data std

        # Mask: only include columns that had non-zero variance
        mask = original_stds > 0
        stds_to_check = stds[mask]

        self.assertTrue(
            np.allclose(stds_to_check, 1, atol=1e-6),
            "Scaled data std should be approximately 1 for non-constant columns."
        )

    # -----------------------------
    # Test 4: K-Means Clustering
    # -----------------------------
    def test_kmeans_labels_length(self):
        """
        Ensure that the number of cluster labels matches the number of samples.
        """
        labels = km.perform_kmeans(self.scaled_data, n_clusters=3)
        self.assertEqual(len(labels), self.scaled_data.shape[0],
                         "Number of cluster labels should match number of samples.")

    def test_kmeans_labels_unique(self):
        """
        Ensure that the number of unique labels equals the number of clusters requested.
        """
        labels = km.perform_kmeans(self.scaled_data, n_clusters=3)
        self.assertEqual(len(np.unique(labels)), 3,
                         "Number of unique labels should equal n_clusters.")

    # -----------------------------
    # Test 5: Visualization
    # -----------------------------
    def test_visualization_runs(self):
        """
        Ensure that the visualization function runs without crashing.
        The test does not check the content of the plot, just that it executes.
        """
        labels = km.perform_kmeans(self.scaled_data, n_clusters=3)
        try:
            km.visualize_clusters(self.scaled_data, labels)
            visualization_success = True
        except Exception:
            visualization_success = False
        self.assertTrue(visualization_success, "Visualization should run without errors.")


# This runs the tests if the file is executed directly
if __name__ == "__main__":
    unittest.main()
