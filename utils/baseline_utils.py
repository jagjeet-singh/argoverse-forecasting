"""This module contains utility functions for all the baselines."""

from collections import OrderedDict
import copy
import math
import os
import pickle as pkl
from typing import Any, Dict, List, Optional, Tuple, Union

from argoverse.map_representation.map_api import ArgoverseMap
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, LinearRing
from shapely.affinity import affine_transform, rotate

from utils.baseline_config import (
    BASELINE_INPUT_FEATURES,
    BASELINE_OUTPUT_FEATURES,
    FEATURE_FORMAT,
)


def get_data(args: Any, baseline_key: str
             ) -> Dict[str, Union[np.ndarray, pd.DataFrame, None]]:
    """Load data from local data_dir.

    Args:
        args (argparse): Arguments to baseline
        baseline_key: Key for obtaining features for the baseline
    Returns:
        data_dict (dict): Dictionary of input/output data and helpers for train/val/test splits

    """
    input_features = BASELINE_INPUT_FEATURES[baseline_key]
    output_features = BASELINE_OUTPUT_FEATURES[baseline_key]
    if args.test_features:
        print("Loading Test data ...")
        test_input, test_output, test_df = load_and_preprocess_data(
            input_features,
            output_features,
            args,
            args.test_features,
            mode="test")
        print("Test Size: {}".format(test_input.shape[0]))
    else:
        test_input, test_output, test_df = [None] * 3

    if args.train_features:
        print("Loading Train data ...")
        train_input, train_output, train_df = load_and_preprocess_data(
            input_features,
            output_features,
            args,
            args.train_features,
            mode="train")
        print("Train Size: {}".format(train_input.shape[0]))
    else:
        train_input, train_output, train_df = [None] * 3

    if args.val_features:
        print("Loading Val data ...")
        val_input, val_output, val_df = load_and_preprocess_data(
            input_features,
            output_features,
            args,
            args.val_features,
            mode="val")
        print("Val Size: {}".format(val_input.shape[0]))
    else:
        val_input, val_output, val_df = [None] * 3

    data_dict = {
        "train_input": train_input,
        "val_input": val_input,
        "test_input": test_input,
        "train_output": train_output,
        "val_output": val_output,
        "test_output": test_output,
        "train_helpers": train_df,
        "val_helpers": val_df,
        "test_helpers": test_df,
    }

    return data_dict


def load_and_preprocess_data(
        input_features: List[str],
        output_features: List[str],
        args: Any,
        feature_file: str,
        mode: str = "train",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load the data and preprocess based on given arguments.

    Args:
        input_features (list of str): Input features for the baseline
        output_features (list of str): Output features for the baseline
        args (argparse): Arguments to runNNBaselines.py/runLSTMBaselines.py
        feature_file: path to the file containing features
        mode (str): train/val/test
    Returns:
        _input: Input to the baseline
        _output: Ground truth 
        df: Helper values useful in visualization and evaluation

    """
    df = pd.read_pickle(feature_file)

    # Normalize if its a non-map baseline
    if not args.use_map and args.normalize:

        print("Normalizing ...")

        # Don't use X,Y as features
        input_feature_idx = [
            FEATURE_FORMAT[feature] for feature in input_features
            if feature != "X" and feature != "Y"
        ]
        output_feature_idx = [
            FEATURE_FORMAT[feature] for feature in output_features
            if feature != "X" and feature != "Y"
        ]

        # Normalize the trajectory
        normalized_traj_arr = get_normalized_traj(df, args)

        # Get other features
        input_features_data = np.stack(
            df["FEATURES"].values)[:, :, input_feature_idx].astype("float")
        output_features_data = np.stack(
            df["FEATURES"].values)[:, :, output_feature_idx].astype("float")

        # Merge normalized trajectory and other features
        input_features_data = np.concatenate(
            (normalized_traj_arr, input_features_data), axis=2)
        output_features_data = np.concatenate(
            (normalized_traj_arr, output_features_data), axis=2)

    else:

        input_feature_idx = [
            FEATURE_FORMAT[feature] for feature in input_features
        ]
        output_feature_idx = [
            FEATURE_FORMAT[feature] for feature in output_features
        ]

        input_features_data = np.stack(
            df["FEATURES"].values)[:, :, input_feature_idx].astype("float")
        output_features_data = np.stack(
            df["FEATURES"].values)[:, :, output_feature_idx].astype("float")

    # If using relative distance instead of absolute
    # Store the first coordinate (reference) of the trajectory to map it back to absolute values later
    if args.use_delta:

        # Get relative distances for all topk centerline candidates
        if args.use_map and mode == "test":

            print("Creating relative distances for candidate centerlines...")

            # Relative candidate distances nt
            candidate_nt_distances = df["CANDIDATE_NT_DISTANCES"].values
            candidate_references = []
            for candidate_nt_dist_i in candidate_nt_distances:
                curr_reference = []
                for curr_candidate_nt in candidate_nt_dist_i:
                    curr_candidate_reference = get_relative_distance(
                        np.expand_dims(curr_candidate_nt, 0), mode, args)
                    curr_candidate_nt = curr_candidate_nt.squeeze()
                    curr_reference.append(curr_candidate_reference.squeeze())
                candidate_references.append(curr_reference)

            df["CANDIDATE_DELTA_REFERENCES"] = candidate_references

        else:

            print("Creating relative distances...")

            # Relative features
            reference = get_relative_distance(input_features_data, mode, args)
            _ = get_relative_distance(output_features_data, mode, args)
            df["DELTA_REFERENCE"] = reference.tolist()

    # Set train and test input/output data
    _input = input_features_data[:, :args.obs_len]

    if mode == "test":
        _output = None
    else:
        _output = output_features_data[:, args.obs_len:]

    return _input, _output, df


def get_relative_distance(data: np.ndarray, mode: str,
                          args: Any) -> np.ndarray:
    """Convert absolute distance to relative distance in place and return the reference (first value).

    Args:
        data (numpy array): Data array of shape (num_tracks x seq_len X num_features). Distances are always the first 2 features
        mode: train/val/test
        args: Arguments passed to the baseline code
    Returns:
        reference (numpy array): First value of the sequence of data with shape (num_tracks x 2). For map based baselines, it will be first n-t distance of the trajectory.

    """
    reference = copy.deepcopy(data[:, 0, :2])

    if mode == "test":
        traj_len = args.obs_len
    else:
        traj_len = args.obs_len + args.pred_len

    for i in range(traj_len - 1, 0, -1):
        data[:, i, :2] = data[:, i, :2] - data[:, i - 1, :2]
    data[:, 0, :] = 0
    return reference


def get_xy_from_nt_seq(nt_seq: np.ndarray,
                       centerlines: List[np.ndarray]) -> np.ndarray:
    """Convert n-t coordinates to x-y, i.e., convert from centerline curvilinear coordinates to map coordinates.

    Args:
        nt_seq (numpy array): Array of shape (num_tracks x seq_len x 2) where last dimension has 'n' (offset from centerline) and 't' (distance along centerline)
        centerlines (list of numpy array): Centerline for each track
    Returns:
        xy_seq (numpy array): Array of shape (num_tracks x seq_len x 2) where last dimension contains coordinates in map frame

    """
    seq_len = nt_seq.shape[1]

    # coordinates obtained by interpolating distances on the centerline
    xy_seq = np.zeros(nt_seq.shape)
    for i in range(nt_seq.shape[0]):
        curr_cl = centerlines[i]
        line_string = LineString(curr_cl)
        for time in range(seq_len):

            # Project nt to xy
            offset_from_cl = nt_seq[i][time][0]
            dist_along_cl = nt_seq[i][time][1]
            x_coord, y_coord = get_xy_from_nt(offset_from_cl, dist_along_cl,
                                              curr_cl)
            xy_seq[i, time, 0] = x_coord
            xy_seq[i, time, 1] = y_coord

    return xy_seq


def get_xy_from_nt(n: float, t: float,
                   centerline: np.ndarray) -> Tuple[float, float]:
    """Convert a single n-t coordinate (centerline curvilinear coordinate) to absolute x-y.

    Args:
        n (float): Offset from centerline
        t (float): Distance along the centerline
        centerline (numpy array): Centerline coordinates
    Returns:
        x1 (float): x-coordinate in map frame
        y1 (float): y-coordinate in map frame

    """
    line_string = LineString(centerline)

    # If distance along centerline is negative, keep it to the start of line
    point_on_cl = line_string.interpolate(
        t) if t > 0 else line_string.interpolate(0)
    local_ls = None

    # Find 2 consective points on centerline such that line joining those 2 points
    # contains point_on_cl
    for i in range(len(centerline) - 1):
        pt1 = centerline[i]
        pt2 = centerline[i + 1]
        ls = LineString([pt1, pt2])
        if ls.distance(point_on_cl) < 1e-8:
            local_ls = ls
            break

    assert local_ls is not None, "XY from N({}) T({}) not computed correctly".format(
        n, t)

    pt1, pt2 = local_ls.coords[:]
    x0, y0 = point_on_cl.coords[0]

    # Determine whether the coordinate lies on left or right side of the line formed by pt1 and pt2
    # Find a point on either side of the line, i.e., (x1_1, y1_1) and (x1_2, y1_2)
    # If the ring formed by (pt1, pt2, (x1_1, y1_1)) is counter clockwise, then it lies on the left

    # Deal with edge cases
    # Vertical
    if pt1[0] == pt2[0]:
        m = 0
        x1_1, x1_2 = x0 + n, x0 - n
        y1_1, y1_2 = y0, y0
    # Horizontal
    elif pt1[1] == pt2[1]:
        m = float("inf")
        x1_1, x1_2 = x0, x0
        y1_1, y1_2 = y0 + n, y0 - n
    # General case
    else:
        ls_slope = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
        m = -1 / ls_slope

        x1_1 = x0 + n / math.sqrt(1 + m**2)
        y1_1 = y0 + m * (x1_1 - x0)
        x1_2 = x0 - n / math.sqrt(1 + m**2)
        y1_2 = y0 + m * (x1_2 - x0)

    # Rings formed by pt1, pt2 and coordinates computed above
    lr1 = LinearRing([pt1, pt2, (x1_1, y1_1)])
    lr2 = LinearRing([pt1, pt2, (x1_2, y1_2)])

    # If ring is counter clockwise
    if lr1.is_ccw:
        x_ccw, y_ccw = x1_1, y1_1
        x_cw, y_cw = x1_2, y1_2
    else:
        x_ccw, y_ccw = x1_2, y1_2
        x_cw, y_cw = x1_1, y1_1

    # If offset is positive, coordinate on the left
    if n > 0:
        x1, y1 = x_ccw, y_ccw
    # Else, coordinate on the right
    else:
        x1, y1 = x_cw, y_cw

    return x1, y1


def viz_predictions(
        input_: np.ndarray,
        output: np.ndarray,
        target: np.ndarray,
        centerlines: np.ndarray,
        city_names: np.ndarray,
        idx=None,
        show: bool = True,
) -> None:
    """Visualize predicted trjectories.

    Args:
        input_ (numpy array): Input Trajectory with shape (num_tracks x obs_len x 2)
        output (numpy array of list): Top-k predicted trajectories, each with shape (num_tracks x pred_len x 2)
        target (numpy array): Ground Truth Trajectory with shape (num_tracks x pred_len x 2)
        centerlines (numpy array of list of centerlines): Centerlines (Oracle/Top-k) for each trajectory
        city_names (numpy array): city names for each trajectory
        show (bool): if True, show

    """
    num_tracks = input_.shape[0]
    obs_len = input_.shape[1]
    pred_len = target.shape[1]

    plt.figure(0, figsize=(8, 7))
    avm = ArgoverseMap()
    for i in range(num_tracks):
        plt.plot(
            input_[i, :, 0],
            input_[i, :, 1],
            color="#ECA154",
            label="Observed",
            alpha=1,
            linewidth=3,
            zorder=15,
        )
        plt.plot(
            input_[i, -1, 0],
            input_[i, -1, 1],
            "o",
            color="#ECA154",
            label="Observed",
            alpha=1,
            linewidth=3,
            zorder=15,
            markersize=9,
        )
        plt.plot(
            target[i, :, 0],
            target[i, :, 1],
            color="#d33e4c",
            label="Target",
            alpha=1,
            linewidth=3,
            zorder=20,
        )
        plt.plot(
            target[i, -1, 0],
            target[i, -1, 1],
            "o",
            color="#d33e4c",
            label="Target",
            alpha=1,
            linewidth=3,
            zorder=20,
            markersize=9,
        )

        for j in range(len(centerlines[i])):
            plt.plot(
                centerlines[i][j][:, 0],
                centerlines[i][j][:, 1],
                "--",
                color="grey",
                alpha=1,
                linewidth=1,
                zorder=0,
            )

        for j in range(len(output[i])):
            plt.plot(
                output[i][j][:, 0],
                output[i][j][:, 1],
                color="#007672",
                label="Predicted",
                alpha=1,
                linewidth=3,
                zorder=15,
            )
            plt.plot(
                output[i][j][-1, 0],
                output[i][j][-1, 1],
                "o",
                color="#007672",
                label="Predicted",
                alpha=1,
                linewidth=3,
                zorder=15,
                markersize=9,
            )
            for k in range(pred_len):
                lane_ids = avm.get_lane_ids_in_xy_bbox(
                    output[i][j][k, 0],
                    output[i][j][k, 1],
                    city_names[i],
                    query_search_range_manhattan=2.5,
                )

        for j in range(obs_len):
            lane_ids = avm.get_lane_ids_in_xy_bbox(
                input_[i, j, 0],
                input_[i, j, 1],
                city_names[i],
                query_search_range_manhattan=2.5,
            )
            [avm.draw_lane(lane_id, city_names[i]) for lane_id in lane_ids]
        for j in range(pred_len):
            lane_ids = avm.get_lane_ids_in_xy_bbox(
                target[i, j, 0],
                target[i, j, 1],
                city_names[i],
                query_search_range_manhattan=2.5,
            )
            [avm.draw_lane(lane_id, city_names[i]) for lane_id in lane_ids]

        plt.axis("equal")
        plt.xticks([])
        plt.yticks([])
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        if show:
            plt.show()


def get_normalized_traj(df: pd.DataFrame, args: Any) -> np.ndarray:
    """Normalize trajectory such that it starts at (0,0) and observed part ends on x-axis.

    Args:
        df (pandas Dataframe): Data for all the tracks
        args: Arguments passed to the baseline code
    Returns:
        normalize_traj_arr (numpy array): Array of shape (num_tracks x seq_len x 2) 
                                          containing normalized trajectory
    Note:
        This also updates the dataframe in-place.

    """
    # Transformation values will be saved in df
    translation = []
    rotation = []

    normalized_traj = []
    x_coord_seq = np.stack(df["FEATURES"].values)[:, :, FEATURE_FORMAT["X"]]
    y_coord_seq = np.stack(df["FEATURES"].values)[:, :, FEATURE_FORMAT["Y"]]

    # Normalize each trajectory
    for i in range(x_coord_seq.shape[0]):
        xy_seq = np.stack((x_coord_seq[i], y_coord_seq[i]), axis=-1)

        start = xy_seq[0]

        # First apply translation
        m = [1, 0, 0, 1, -start[0], -start[1]]
        ls = LineString(xy_seq)

        # Now apply rotation, taking care of edge cases
        ls_offset = affine_transform(ls, m)
        end = ls_offset.coords[args.obs_len - 1]
        if end[0] == 0 and end[1] == 0:
            angle = 0.0
        elif end[0] == 0:
            angle = -90.0 if end[1] > 0 else 90.0
        elif end[1] == 0:
            angle = 0.0 if end[0] > 0 else 180.0
        else:
            angle = math.degrees(math.atan(end[1] / end[0]))
            if (end[0] > 0 and end[1] > 0) or (end[0] > 0 and end[1] < 0):
                angle = -angle
            else:
                angle = 180.0 - angle

        # Rotate the trajetory
        ls_rotate = rotate(ls_offset, angle, origin=(0, 0)).coords[:]

        # Normalized trajectory
        norm_xy = np.array(ls_rotate)

        # Update the containers
        normalized_traj.append(norm_xy)
        translation.append(m)
        rotation.append(angle)

    # Update the dataframe and return the normalized trajectory
    normalize_traj_arr = np.stack(normalized_traj)
    df["TRANSLATION"] = translation
    df["ROTATION"] = rotation
    return normalize_traj_arr


def normalized_to_map_coordinates(coords: np.ndarray,
                                  translation: List[List[float]],
                                  rotation: List[float]) -> np.ndarray:
    """Denormalize trajectory to bring it back to map frame.

    Args:
        coords (numpy array): Array of shape (num_tracks x seq_len x 2) containing normalized coordinates
        translation (list): Translation matrix used in normalizing trajectories
        rotation (list): Rotation angle used in normalizing trajectories 
    Returns:
        _ (numpy array: Array of shape (num_tracks x seq_len x 2) containing coordinates in map frame

    """
    abs_coords = []
    for i in range(coords.shape[0]):
        ls = LineString(coords[i])

        # Rotate
        ls_rotate = rotate(ls, -rotation[i], origin=(0, 0))

        # Translate
        M_inv = [1, 0, 0, 1, -translation[i][4], -translation[i][5]]

        ls_offset = affine_transform(ls_rotate, M_inv).coords[:]
        abs_coords.append(ls_offset)

    return np.array(abs_coords)


def get_abs_traj(
        input_: np.ndarray,
        output: np.ndarray,
        args: Any,
        helpers: Dict[str, Any],
        start_idx: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get absolute trajectory reverting all the transformations.

    Args:
        input_ (numpy array): Input Trajectory with shape (num_tracks x obs_len x 2)
        output (numpy array): Predicted Trajectory with shape (num_tracks x pred_len x 2)
        args (Argparse): Config parameters
        helpers (dict):Data helpers
        start_id (int): Start index of the current batch (used in joblib). If None, then no batching.
    Returns:            
        input_ (numpy array): Input Trajectory in map frame with shape (num_tracks x obs_len x 2)
        output (numpy array): Predicted Trajectory in map frame with shape (num_tracks x pred_len x 2)

    """
    obs_len = input_.shape[1]
    pred_len = output.shape[1]

    if start_idx is None:
        s = 0
        e = input_.shape[0]
    else:
        print(f"Abs Traj Done {start_idx}/{input_.shape[0]}")
        s = start_idx
        e = start_idx + args.joblib_batch_size

    input_ = input_.copy()[s:e]
    output = output.copy()[s:e]

    # Convert relative to absolute
    if args.use_delta:
        reference = helpers["REFERENCE"].copy()[s:e]
        input_[:, 0, :2] = reference
        for i in range(1, obs_len):
            input_[:, i, :2] = input_[:, i, :2] + input_[:, i - 1, :2]

        output[:, 0, :2] = output[:, 0, :2] + input_[:, -1, :2]
        for i in range(1, pred_len):
            output[:, i, :2] = output[:, i, :2] + output[:, i - 1, :2]

    # Convert centerline frame (n,t) to absolute frame (x,y)
    if args.use_map:
        centerlines = helpers["CENTERLINE"].copy()[s:e]
        input_[:, :, :2] = get_xy_from_nt_seq(input_[:, :, :2], centerlines)
        output[:, :, :2] = get_xy_from_nt_seq(output[:, :, :2], centerlines)

    # Denormalize trajectory
    elif args.normalize and not args.use_map:
        translation = helpers["TRANSLATION"].copy()[s:e]
        rotation = helpers["ROTATION"].copy()[s:e]
        input_[:, :, :2] = normalized_to_map_coordinates(
            input_[:, :, :2], translation, rotation)
        output[:, :, :2] = normalized_to_map_coordinates(
            output[:, :, :2], translation, rotation)
    return input_, output


def get_model(
        regressor: Any,
        train_input: np.ndarray,
        train_output: np.ndarray,
        args: Any,
        pred_horizon: int,
) -> Any:
    """Get the trained model after running grid search or load a saved one.

    Args:
        regressor: Nearest Neighbor regressor class instance 
        train_input: Input to the model
        train_output: Ground truth for the model
        args: Arguments passed to the baseline
        pred_horizon: Prediction Horizon

    Returns:
        grid_search: sklearn GridSearchCV object

    """
    # Load model
    if args.test:

        # Load a trained model
        with open(args.model_path, "rb") as f:
            grid_search = pkl.load(f)
        print(f"## Loaded {args.model_path} ....")

    else:

        train_num_tracks = train_input.shape[0]

        # Flatten to (num_tracks x feature_size)
        train_output_curr = train_output[:, :pred_horizon, :].reshape(
            (train_num_tracks, pred_horizon * 2), order="F")

        # Run grid search for hyper parameter tuning
        grid_search = regressor.run_grid_search(train_input, train_output_curr)
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        with open(args.model_path, "wb") as f:
            pkl.dump(grid_search, f)
        print(f"Trained model saved at... {args.model_path}")

    return grid_search


def merge_saved_traj(batched_dir: str, merged_file_path: str):
    """Load saved trajectories, merge them, save the merged one, delete the individual ones.

    Args:
        batched_dir: Directory where forecasted trajectories for all the batches are saved
        merged_file_path: Path to the pickle file where merged file is to be saved.
    Note: batched_dir should only contain the files that are to be merged

    """
    file_names = os.listdir(batched_dir)
    forecasted_trajectories = {}
    for fn in file_names:
        file_path = f"{batched_dir}/{fn}"
        with open(file_path, "rb") as f:
            traj = pkl.load(f)
        forecasted_trajectories = {**forecasted_trajectories, **traj}
        os.remove(file_path)
    with open(merged_file_path, "wb") as f:
        pkl.dump(forecasted_trajectories, f)


def get_test_data_dict_subset(
        data_dict: Dict[str, Union[np.ndarray, None]],
        args: Any) -> Dict[int, Dict[str, Union[np.ndarray, None]]]:
    """Get test subset from data dict. Useful when used with joblib as we don't need to pass the entire data_dict to all the batches.

    Args:
        data_dict: Data dictionary containing all the data
        args: Arguments passed to the baseline

    Returns:
        test_data_dict_batches: test data subsets. key is the start index of the joblib batch
                                and value is the subset of test data corresponding to that batch.

    """
    test_size = data_dict["test_input"].shape[0]
    test_data_dict_batches = {}
    for i in range(0, test_size, args.joblib_batch_size):
        new_dict = {}
        for k, v in data_dict.items():
            if k in ["test_input", "test_helpers"]:
                new_dict[k] = v[i:i + args.joblib_batch_size]
        test_data_dict_batches[i] = new_dict
    return test_data_dict_batches


def validate_args(args: Any) -> bool:
    """Validate the arguments passed to the baseline.

    Args:
        args: Arguments to the baselines.

    Returns:
        success: True if args valid.

    """
    success = True
    if args.normalize and args.use_map:
        print(
            "[ARGS ERROR]: normalize and use_map cannot be used simultaneously."
        )
        success = False
    if args.use_social and args.use_map:
        print(
            "[ARGS ERROR]: The code currently does not support use_social and use_map simultaneously."
        )
        success = False
    if args.obs_len > 20:
        print("[ARGS ERROR]: obs_len cannot be more than 20.")
        success = False
    if args.pred_len > 30:
        print("[ARGS ERROR]: pred_len cannot be more than 30.")
        success = False
    return success
