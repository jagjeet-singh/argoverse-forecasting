"""This module is used for computing social and map features for motion forecasting baselines.

Example usage:
    $ python compute_features.py --data_dir ~/val/data 
        --feature_dir ~/val/features --mode val
"""

import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Tuple

import argparse
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import pickle as pkl

from utils.baseline_config import RAW_DATA_FORMAT, _FEATURES_SMALL_SIZE
from utils.map_features_utils import MapFeaturesUtils
from utils.social_features_utils import SocialFeaturesUtils


def parse_arguments() -> Any:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="",
        type=str,
        help="Directory where the sequences (csv files) are saved",
    )
    parser.add_argument(
        "--feature_dir",
        default="",
        type=str,
        help="Directory where the computed features are to be saved",
    )
    parser.add_argument("--mode",
                        required=True,
                        type=str,
                        help="train/val/test")
    parser.add_argument(
        "--batch_size",
        default=100,
        type=int,
        help="Batch size for parallel computation",
    )
    parser.add_argument("--obs_len",
                        default=20,
                        type=int,
                        help="Observed length of the trajectory")
    parser.add_argument("--pred_len",
                        default=30,
                        type=int,
                        help="Prediction Horizon")
    parser.add_argument("--small",
                        action="store_true",
                        help="If true, a small subset of data is used.")
    return parser.parse_args()


def load_seq_save_features(
        start_idx: int,
        sequences: List[str],
        save_dir: str,
        map_features_utils_instance: MapFeaturesUtils,
        social_features_utils_instance: SocialFeaturesUtils,
) -> None:
    """Load sequences, compute features, and save them.
    
    Args:
        start_idx : Starting index of the current batch
        sequences : Sequence file names
        save_dir: Directory where features for the current batch are to be saved
        map_features_utils_instance: MapFeaturesUtils instance
        social_features_utils_instance: SocialFeaturesUtils instance

    """
    count = 0
    args = parse_arguments()
    data = []

    # Enumerate over the batch starting at start_idx
    for seq in sequences[start_idx:start_idx + args.batch_size]:

        if not seq.endswith(".csv"):
            continue

        file_path = f"{args.data_dir}/{seq}"
        seq_id = int(seq.split(".")[0])

        # Compute social and map features
        features, map_feature_helpers = compute_features(
            file_path, map_features_utils_instance,
            social_features_utils_instance)
        count += 1
        data.append([
            seq_id,
            features,
            map_feature_helpers["CANDIDATE_CENTERLINES"],
            map_feature_helpers["ORACLE_CENTERLINE"],
            map_feature_helpers["CANDIDATE_NT_DISTANCES"],
        ])

        print(
            f"{args.mode}:{count}/{args.batch_size} with start {start_idx} and end {start_idx + args.batch_size}"
        )

    data_df = pd.DataFrame(
        data,
        columns=[
            "SEQUENCE",
            "FEATURES",
            "CANDIDATE_CENTERLINES",
            "ORACLE_CENTERLINE",
            "CANDIDATE_NT_DISTANCES",
        ],
    )

    # Save the computed features for all the sequences in the batch as a single file
    os.makedirs(save_dir, exist_ok=True)
    data_df.to_pickle(
        f"{save_dir}/forecasting_features_{args.mode}_{start_idx}_{start_idx + args.batch_size}.pkl"
    )


def compute_features(
        seq_path: str,
        map_features_utils_instance: MapFeaturesUtils,
        social_features_utils_instance: SocialFeaturesUtils,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute social and map features for the sequence.

    Args:
        seq_path (str): file path for the sequence whose features are to be computed.
        map_features_utils_instance: MapFeaturesUtils instance.
        social_features_utils_instance: SocialFeaturesUtils instance.
    Returns:
        merged_features (numpy array): SEQ_LEN x NUM_FEATURES
        map_feature_helpers (dict): Dictionary containing helpers for map features

    """
    args = parse_arguments()
    df = pd.read_csv(seq_path, dtype={"TIMESTAMP": str})

    # Get social and map features for the agent
    agent_track = df[df["OBJECT_TYPE"] == "AGENT"].values

    # Social features are computed using only the observed trajectory
    social_features = social_features_utils_instance.compute_social_features(
        df, agent_track, args.obs_len, args.obs_len + args.pred_len,
        RAW_DATA_FORMAT)

    # agent_track will be used to compute n-t distances for future trajectory,
    # using centerlines obtained from observed trajectory
    map_features, map_feature_helpers = map_features_utils_instance.compute_map_features(
        agent_track,
        args.obs_len,
        args.obs_len + args.pred_len,
        RAW_DATA_FORMAT,
        args.mode,
    )

    # Combine social and map features

    # If track is of OBS_LEN (i.e., if it's in test mode), use agent_track of full SEQ_LEN,
    # But keep (OBS_LEN+1) to (SEQ_LEN) indexes having None values
    if agent_track.shape[0] == args.obs_len:
        agent_track_seq = np.full(
            (args.obs_len + args.pred_len, agent_track.shape[1]), None)
        agent_track_seq[:args.obs_len] = agent_track
        merged_features = np.concatenate(
            (agent_track_seq, social_features, map_features), axis=1)
    else:
        merged_features = np.concatenate(
            (agent_track, social_features, map_features), axis=1)

    return merged_features, map_feature_helpers


def merge_saved_features(batch_save_dir: str) -> None:
    """Merge features saved by parallel jobs.

    Args:
        batch_save_dir: Directory where features for all the batches are saved.

    """
    args = parse_arguments()
    feature_files = os.listdir(batch_save_dir)
    all_features = []
    for feature_file in feature_files:
        if not feature_file.endswith(".pkl") or args.mode not in feature_file:
            continue
        file_path = f"{batch_save_dir}/{feature_file}"
        df = pd.read_pickle(file_path)
        all_features.append(df)

        # Remove the batch file
        os.remove(file_path)

    all_features_df = pd.concat(all_features, ignore_index=True)

    # Save the features for all the sequences into a single file
    all_features_df.to_pickle(
        f"{args.feature_dir}/forecasting_features_{args.mode}.pkl")


if __name__ == "__main__":
    """Load sequences and save the computed features."""
    args = parse_arguments()

    start = time.time()

    map_features_utils_instance = MapFeaturesUtils()
    social_features_utils_instance = SocialFeaturesUtils()

    sequences = os.listdir(args.data_dir)
    temp_save_dir = tempfile.mkdtemp()

    num_sequences = _FEATURES_SMALL_SIZE if args.small else len(sequences)

    Parallel(n_jobs=-2)(delayed(load_seq_save_features)(
        i,
        sequences,
        temp_save_dir,
        map_features_utils_instance,
        social_features_utils_instance,
    ) for i in range(0, num_sequences, args.batch_size))
    merge_saved_features(temp_save_dir)
    shutil.rmtree(temp_save_dir)

    print(
        f"Feature computation for {args.mode} set completed in {(time.time()-start)/60.0} mins"
    )
