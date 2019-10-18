"""This module contains utility functions for training and testing Nearest Neighbor based baselines."""

import os
import shutil
import tempfile
from typing import Any, Dict, List, Union

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

import utils.baseline_utils as baseline_utils

PREDICTION_HORIZONS = [30]


class Regressor:
    """K-NN Regressor class for training and inference."""
    def run_grid_search(self, x: np.ndarray, y: np.ndarray) -> GridSearchCV:
        """Run grid search to find best hyperparameters.

        Args:
            x: Train input
            y: Train output
        Returns:
            GridSearchCV object

        """
        estimators = [
            ("preprocessing", preprocessing.StandardScaler()),
            ("reduce_dim", PCA()),
            ("regressor", KNeighborsRegressor()),
        ]
        pipe = Pipeline(estimators)

        param_grid = dict(
            preprocessing=[None, preprocessing.StandardScaler()],
            reduce_dim=[None, PCA(32)],
            regressor__n_neighbors=[1, 8, 16],
            regressor__weights=["distance"],
            regressor__n_jobs=[-2],
        )

        custom_scorer = make_scorer(mean_squared_error,
                                    greater_is_better=False)

        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=custom_scorer,
            cv=3,
            return_train_score=False,
            verbose=1,
        )

        grid_search.fit(x, y)

        return grid_search

    def train_and_infer_map(
            self,
            train_input: np.ndarray,
            train_output: np.ndarray,
            test_helpers: pd.DataFrame,
            num_features: int,
            args: Any,
    ) -> None:
        """Train and test the model on different prediction horizons for map based Nearest Neighbors.

        Args:
            train_input (numpy array): Train input data 
            train_output (numpy array): Train ground truth data 
            test_helpers (pandas Dataframe): Test map helpers 
            num_features: Number of input features
            args: Arguments passed to runNNBaselines.py

        """
        # Create a temporary directory where forecasted trajectories for all the batches will be saved
        temp_save_dir = tempfile.mkdtemp()

        print(
            f"Forecasted trajectories will be saved in {args.traj_save_path} ..."
        )

        # Train and Test inputs for NN
        train_num_tracks = train_input.shape[0]
        train_input = train_input.reshape(
            (train_num_tracks, args.obs_len * num_features), order="F")

        # Get test centerline
        test_centerlines = test_helpers["CANDIDATE_CENTERLINES"].values
        test_nt = test_helpers["CANDIDATE_NT_DISTANCES"].values
        test_references = test_helpers["CANDIDATE_DELTA_REFERENCES"].values
        test_seq_ids = test_helpers["SEQUENCE"].values

        test_num_tracks = test_nt.shape[0]

        for curr_pred_horizon in PREDICTION_HORIZONS:

            grid_search = baseline_utils.get_model(self, train_input,
                                                   train_output, args,
                                                   curr_pred_horizon)

            print("Model obtained, now starting inference ...")

            Parallel(
                n_jobs=-2, verbose=5
            )(delayed(self.infer_and_save_traj_map)(
                grid_search,
                train_output,
                test_nt[i:min(i + args.joblib_batch_size, test_num_tracks)],
                test_centerlines[i:min(i +
                                       args.joblib_batch_size, test_num_tracks
                                       )],
                test_references[i:min(i +
                                      args.joblib_batch_size, test_num_tracks
                                      )],
                test_seq_ids[i:min(i +
                                   args.joblib_batch_size, test_num_tracks)],
                start_idx=i,
                args=args,
                num_features=num_features,
                horizon=curr_pred_horizon,
                save_dir=temp_save_dir,
            ) for i in range(0, test_num_tracks, args.joblib_batch_size))
        baseline_utils.merge_saved_traj(temp_save_dir, args.traj_save_path)
        shutil.rmtree(temp_save_dir)

    def train_and_infer_absolute(
            self,
            train_input: np.ndarray,
            train_output: np.ndarray,
            test_input: np.ndarray,
            test_helpers: pd.DataFrame,
            num_features: int,
            args: Any,
    ) -> None:
        """Train and test the model on different prediction horizons for non-map baselines.

        Args:
            train_input (numpy array): Train input data 
            train_output (numpy array): Train ground truth data 
            test_input (numpy array): Test input data  
            test_helpers (pandas Dataframe): Test map helpers 
            num_features: Number of input features
            args (Argparse): Config parameters

        """
        # Create a temporary directory where forecasted trajectories for all the batches will be saved
        temp_save_dir = tempfile.mkdtemp()

        print(
            f"Forecasted trajectories will be saved in {args.traj_save_path} ..."
        )

        # Train and Test inputs for NN
        test_num_tracks = test_input.shape[0]
        test_input = test_input.reshape(
            (test_num_tracks, args.obs_len * num_features), order="F")

        train_num_tracks = train_input.shape[0]
        train_input = train_input.reshape(
            (train_num_tracks, args.obs_len * num_features), order="F")

        test_references = test_helpers["DELTA_REFERENCE"].values
        test_translation = (test_helpers["TRANSLATION"].values
                            if args.normalize else None)
        test_rotation = test_helpers[
            "ROTATION"].values if args.normalize else None
        test_seq_ids = test_helpers["SEQUENCE"].values

        # Run experiments over different prediction horizons
        for curr_pred_horizon in PREDICTION_HORIZONS:

            grid_search = baseline_utils.get_model(self, train_input,
                                                   train_output, args,
                                                   curr_pred_horizon)

            print("Model obtained, now starting inference ...")

            Parallel(
                n_jobs=-2, verbose=5
            )(delayed(self.infer_and_save_traj_absolute)(
                grid_search,
                train_output,
                test_input[i:min(i + args.joblib_batch_size, test_num_tracks)],
                test_references[i:min(i +
                                      args.joblib_batch_size, test_num_tracks
                                      )] if args.use_delta else None,
                test_seq_ids[i:min(i +
                                   args.joblib_batch_size, test_num_tracks)],
                test_translation[i:min(i +
                                       args.joblib_batch_size, test_num_tracks
                                       )] if args.normalize else None,
                test_rotation[i:min(i + args.joblib_batch_size, test_num_tracks
                                    )] if args.normalize else None,
                start_idx=i,
                args=args,
                num_features=num_features,
                horizon=curr_pred_horizon,
                save_dir=temp_save_dir,
            ) for i in range(0, test_num_tracks, args.joblib_batch_size))
        baseline_utils.merge_saved_traj(temp_save_dir, args.traj_save_path)
        shutil.rmtree(temp_save_dir)

    def infer_and_save_traj_map(
            self,
            grid_search: Any,
            train_output: np.ndarray,
            test_nt: np.ndarray,
            test_centerlines: np.ndarray,
            test_references: np.ndarray,
            test_seq_ids: np.ndarray,
            start_idx: int,
            args: Any,
            num_features: int,
            horizon: int,
            save_dir: str,
    ) -> np.ndarray:
        """Map-based baselines inference. This function does the inference based on the given model, and saves the forecasted trajectories.

        Args:
            grid_search: GridSearchCV object,
            train_output: Train output of shape [num_tracks, pred_len, num_features]
            test_nt: Candidate trajectories in centerline curvilinear coordinates,
            test_centerlines: Candidate centerlines,
            test_references: References for reverting delta transformation of candidate trajectories,
            test_seq_ids: csv name for the sequence,
            start_idx: start_idx for current joblib batch,
            args: Arguments passed to the baseline script,
            num_features: Number of features,
            horizon: Prediction Horizon, 
            save_dir: Directory where forecasted trajectories are to be saved 
        Returns:
            forecasted_trajectories: Forecasted trajectories 

        """
        test_num_tracks = test_nt.shape[0]
        print(f"Map-based Inference currently at index {start_idx} ...")

        forecasted_trajectories = {}

        # Predict for each trajectory
        for i in range(test_num_tracks):

            # Helpers for current track
            test_nt_i = test_nt[i]
            test_cl_i = test_centerlines[i]
            test_references_i = test_references[i]

            curr_forecasted_trajectories = []

            # Predict using each of candidate centerlines
            for (
                    test_nt_i_curr_candidate,
                    test_cl_i_curr_candidate,
                    test_ref_i_curr_candidate,
            ) in zip(test_nt_i, test_cl_i, test_references_i):
                test_input = test_nt_i_curr_candidate[:args.
                                                      obs_len, :].reshape(
                                                          (1,
                                                           2 * args.obs_len),
                                                          order="F")

                # Preprocess and get neighbors
                pipeline_steps = grid_search.best_estimator_.named_steps.keys()
                preprocessed_test_input = np.copy(test_input)
                for step in pipeline_steps:
                    curr_step = grid_search.best_estimator_.named_steps[step]

                    # Get neighbors
                    if step == "regressor":
                        neigh_idx = curr_step.kneighbors(
                            preprocessed_test_input,
                            return_distance=False,
                            n_neighbors=args.n_neigh,
                        )

                    # Preprocess
                    else:
                        if curr_step is not None:
                            print(preprocessed_test_input.shape)
                            preprocessed_test_input = curr_step.transform(
                                preprocessed_test_input)

                y_pred = train_output[neigh_idx][
                    0, :, :horizon, :]  # num_neighbors x curr_pred_len x 2
                test_input = np.repeat(test_input,
                                       repeats=args.n_neigh,
                                       axis=0)

                abs_helpers = {}
                abs_helpers["CENTERLINE"] = [
                    test_cl_i_curr_candidate for i in range(args.n_neigh)
                ]
                if args.use_delta:

                    abs_helpers["REFERENCE"] = np.array([
                        test_ref_i_curr_candidate for i in range(args.n_neigh)
                    ])

                # Convert trajectory to map frame
                abs_input, abs_output = baseline_utils.get_abs_traj(
                    test_input.copy().reshape((-1, args.obs_len, num_features),
                                              order="F"),
                    y_pred.copy(),
                    args,
                    helpers=abs_helpers,
                )
                curr_forecasted_trajectories.extend(abs_output)
            forecasted_trajectories[
                test_seq_ids[i]] = curr_forecasted_trajectories
        with open(f"{save_dir}/{start_idx}.pkl", "wb") as f:
            pkl.dump(forecasted_trajectories, f)

    def infer_and_save_traj_absolute(
            self,
            grid_search: Any,
            train_output: np.ndarray,
            test_input: np.ndarray,
            test_references: Union[np.ndarray, None],
            test_seq_ids: np.ndarray,
            test_translation: Union[np.ndarray, None],
            test_rotation: Union[np.ndarray, None],
            start_idx: int,
            args: Any,
            num_features: int,
            horizon: int,
            save_dir: str,
    ) -> None:
        """Non-map baselines inference. This function does the inference based on the given model, and saves the forecasted trajectories.

        Args:
            grid_search: GridSearchCV object
            train_output: Train output
            test_input: Test input
            test_references: References for reverting delta transformation of candidate trajectories,
            test_seq_ids: csv name for the sequence,
            test_translation: Translation used for normalizing trajectories,
            test_rotation: Rotation used for normalizing trajectories,
            start_idx: start_idx for current joblib batch,
            args: Arguments passed to the baseline script,
            num_features: Number of features,
            horizon: Prediction Horizon, 
            save_dir: Directory where forecasted trajectories are to be saved      

        """
        test_num_tracks = test_input.shape[0]
        print(f"Absolute Inference currently at index {start_idx} ...")

        forecasted_trajectories = {}

        # Preprocess and get neighbors
        pipeline_steps = grid_search.best_estimator_.named_steps.keys()
        preprocessed_test_input = np.copy(test_input)
        for step in pipeline_steps:
            curr_step = grid_search.best_estimator_.named_steps[step]

            # Get neighbors
            if step == "regressor":
                neigh_idx = curr_step.kneighbors(
                    preprocessed_test_input,
                    return_distance=False,
                    n_neighbors=args.n_neigh,
                )

            # Preprocess
            else:
                if curr_step is not None:
                    preprocessed_test_input = curr_step.transform(
                        preprocessed_test_input)

        # Predict for each trajectory
        for i in range(test_num_tracks):

            curr_test_input = np.repeat(test_input[i],
                                        repeats=args.n_neigh,
                                        axis=0)
            neigh_output = train_output[neigh_idx][
                i, :, :horizon, :]  # num_neigh x curr_pred_len x 2

            abs_helpers = {}
            if args.use_delta:
                abs_helpers["REFERENCE"] = np.array(
                    [test_references[i] for _ in range(args.n_neigh)])
            if args.normalize:
                abs_helpers["TRANSLATION"] = np.array(
                    [test_translation[i] for _ in range(args.n_neigh)])
                abs_helpers["ROTATION"] = np.array(
                    [test_rotation[i] for _ in range(args.n_neigh)])

            # Convert trajectory to map frame
            abs_input, abs_output = baseline_utils.get_abs_traj(
                curr_test_input.copy().reshape(
                    (-1, args.obs_len, num_features), order="F"),
                neigh_output.copy(),
                args,
                helpers=abs_helpers,
            )
            forecasted_trajectories[test_seq_ids[i]] = abs_output
        with open(f"{save_dir}/{start_idx}.pkl", "wb") as f:
            pkl.dump(forecasted_trajectories, f)
