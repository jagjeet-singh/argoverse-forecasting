"""This module is used for computing social features for motion forecasting baselines."""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.baseline_config import (
    PADDING_TYPE,
    STATIONARY_THRESHOLD,
    VELOCITY_THRESHOLD,
    EXIST_THRESHOLD,
    DEFAULT_MIN_DIST_FRONT_AND_BACK,
    NEARBY_DISTANCE_THRESHOLD,
    FRONT_OR_BACK_OFFSET_THRESHOLD,
)


class SocialFeaturesUtils:
    """Utils class for computation of social features."""
    def __init__(self):
        """Initialize class."""
        self.PADDING_TYPE = PADDING_TYPE
        self.STATIONARY_THRESHOLD = STATIONARY_THRESHOLD
        self.VELOCITY_THRESHOLD = VELOCITY_THRESHOLD
        self.EXIST_THRESHOLD = EXIST_THRESHOLD
        self.DEFAULT_MIN_DIST_FRONT_AND_BACK = DEFAULT_MIN_DIST_FRONT_AND_BACK
        self.NEARBY_DISTANCE_THRESHOLD = NEARBY_DISTANCE_THRESHOLD

    def compute_velocity(self, track_df: pd.DataFrame) -> List[float]:
        """Compute velocities for the given track.

        Args:
            track_df (pandas Dataframe): Data for the track
        Returns:
            vel (list of float): Velocity at each timestep

        """
        x_coord = track_df["X"].values
        y_coord = track_df["Y"].values
        timestamp = track_df["TIMESTAMP"].values
        vel_x, vel_y = zip(*[(
            x_coord[i] - x_coord[i - 1] /
            (float(timestamp[i]) - float(timestamp[i - 1])),
            y_coord[i] - y_coord[i - 1] /
            (float(timestamp[i]) - float(timestamp[i - 1])),
        ) for i in range(1, len(timestamp))])
        vel = [np.sqrt(x**2 + y**2) for x, y in zip(vel_x, vel_y)]

        return vel

    def get_is_track_stationary(self, track_df: pd.DataFrame) -> bool:
        """Check if the track is stationary.

        Args:
            track_df (pandas Dataframe): Data for the track
        Return:
            _ (bool): True if track is stationary, else False 

        """
        vel = self.compute_velocity(track_df)
        sorted_vel = sorted(vel)
        threshold_vel = sorted_vel[self.STATIONARY_THRESHOLD]
        return True if threshold_vel < self.VELOCITY_THRESHOLD else False

    def fill_track_lost_in_middle(
            self,
            track_array: np.ndarray,
            seq_timestamps: np.ndarray,
            raw_data_format: Dict[str, int],
    ) -> np.ndarray:
        """Handle the case where the object exited and then entered the frame but still retains the same track id. It'll be a rare case.

        Args:
            track_array (numpy array): Padded data for the track
            seq_timestamps (numpy array): All timestamps in the sequence
            raw_data_format (Dict): Format of the sequence
        Returns:
            filled_track (numpy array): Track data filled with missing timestamps

        """
        curr_idx = 0
        filled_track = np.empty((0, track_array.shape[1]))
        for timestamp in seq_timestamps:
            filled_track = np.vstack((filled_track, track_array[curr_idx]))
            if timestamp in track_array[:, raw_data_format["TIMESTAMP"]]:
                curr_idx += 1
        return filled_track

    def pad_track(
            self,
            track_df: pd.DataFrame,
            seq_timestamps: np.ndarray,
            obs_len: int,
            raw_data_format: Dict[str, int],
    ) -> np.ndarray:
        """Pad incomplete tracks.

        Args:
            track_df (Dataframe): Dataframe for the track
            seq_timestamps (numpy array): All timestamps in the sequence
            obs_len (int): Length of observed trajectory
            raw_data_format (Dict): Format of the sequence
        Returns:
                padded_track_array (numpy array): Track data padded in front and back

        """
        track_vals = track_df.values
        track_timestamps = track_df["TIMESTAMP"].values

        # start and index of the track in the sequence
        start_idx = np.where(seq_timestamps == track_timestamps[0])[0][0]
        end_idx = np.where(seq_timestamps == track_timestamps[-1])[0][0]

        # Edge padding in front and rear, i.e., repeat the first and last coordinates
        if self.PADDING_TYPE == "REPEAT":
            padded_track_array = np.pad(track_vals,
                                        ((start_idx, obs_len - end_idx - 1),
                                         (0, 0)), "edge")
            if padded_track_array.shape[0] < obs_len:
                padded_track_array = self.fill_track_lost_in_middle(
                    padded_track_array, seq_timestamps, raw_data_format)

        # Overwrite the timestamps in padded part
        for i in range(padded_track_array.shape[0]):
            padded_track_array[i, 0] = seq_timestamps[i]
        return padded_track_array

    def filter_tracks(self, seq_df: pd.DataFrame, obs_len: int,
                      raw_data_format: Dict[str, int]) -> np.ndarray:
        """Pad tracks which don't last throughout the sequence. Also, filter out non-relevant tracks.

        Args:
            seq_df (pandas Dataframe): Dataframe containing all the tracks in the sequence
            obs_len (int): Length of observed trajectory
            raw_data_format (Dict): Format of the sequence
        Returns:
            social_tracks (numpy array): Array of relevant tracks

        """
        social_tracks = np.empty((0, obs_len, len(raw_data_format)))

        # Timestamps in the sequence
        seq_timestamps = np.unique(seq_df["TIMESTAMP"].values)

        # Track groups
        df_groups = seq_df.groupby("TRACK_ID")
        for group_name, group_data in df_groups:

            # Check if the track is long enough
            if len(group_data) < self.EXIST_THRESHOLD:
                continue

            # Skip if agent track
            if group_data["OBJECT_TYPE"].iloc[0] == "AGENT":
                continue

            # Check if the track is stationary
            if self.get_is_track_stationary(group_data):
                continue

            padded_track_array = self.pad_track(group_data, seq_timestamps,
                                                obs_len,
                                                raw_data_format).reshape(
                                                    (1, obs_len, -1))
            social_tracks = np.vstack((social_tracks, padded_track_array))

        return social_tracks

    def get_is_front_or_back(
            self,
            track: np.ndarray,
            neigh_x: float,
            neigh_y: float,
            raw_data_format: Dict[str, int],
    ) -> Optional[str]:
        """Check if the neighbor is in front or back of the track.

        Args:
            track (numpy array): Track data
            neigh_x (float): Neighbor x coordinate
            neigh_y (float): Neighbor y coordinate
        Returns:
            _ (str): 'front' if in front, 'back' if in back

        """
        # We don't have heading information. So we need at least 2 coordinates to determine that.
        # Here, front and back is determined wrt to last 2 coordinates of the track
        x2 = track[-1, raw_data_format["X"]]
        y2 = track[-1, raw_data_format["Y"]]

        # Keep taking previous coordinate until first distinct coordinate is found.
        idx1 = track.shape[0] - 2
        while idx1 > -1:
            x1 = track[idx1, raw_data_format["X"]]
            y1 = track[idx1, raw_data_format["Y"]]
            if x1 != x2 or y1 != y2:
                break
            idx1 -= 1

        # If all the coordinates in the track are the same, there's no way to find front/back
        if idx1 < 0:
            return None

        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])
        p3 = np.array([neigh_x, neigh_y])
        proj_dist = np.abs(np.cross(p2 - p1,
                                    p1 - p3)) / np.linalg.norm(p2 - p1)

        # Interested in only those neighbors who are not far away from the direction of travel
        if proj_dist < FRONT_OR_BACK_OFFSET_THRESHOLD:

            dist_from_end_of_track = np.sqrt(
                (track[-1, raw_data_format["X"]] - neigh_x)**2 +
                (track[-1, raw_data_format["Y"]] - neigh_y)**2)
            dist_from_start_of_track = np.sqrt(
                (track[0, raw_data_format["X"]] - neigh_x)**2 +
                (track[0, raw_data_format["Y"]] - neigh_y)**2)
            dist_start_end = np.sqrt((track[-1, raw_data_format["X"]] -
                                      track[0, raw_data_format["X"]])**2 +
                                     (track[-1, raw_data_format["Y"]] -
                                      track[0, raw_data_format["Y"]])**2)

            return ("front"
                    if dist_from_end_of_track < dist_from_start_of_track
                    and dist_from_start_of_track > dist_start_end else "back")

        else:
            return None

    def get_min_distance_front_and_back(
            self,
            agent_track: np.ndarray,
            social_tracks: np.ndarray,
            obs_len: int,
            raw_data_format: Dict[str, int],
            viz=False,
    ) -> np.ndarray:
        """Get minimum distance of the tracks in front and in back.

        Args:
            agent_track (numpy array): Data for the agent track
            social_tracks (numpy array): Array of relevant tracks
            obs_len (int): Length of the observed trajectory
            raw_data_format (Dict): Format of the sequence
            viz (bool): Visualize tracks
        Returns:
            min_distance_front_and_back (numpy array): obs_len x 2, minimum front and back distances

        """
        min_distance_front_and_back = np.full(
            (obs_len, 2), self.DEFAULT_MIN_DIST_FRONT_AND_BACK)

        # Compute distances for each timestep in the sequence
        for i in range(obs_len):

            # Agent coordinates
            agent_x, agent_y = (
                agent_track[i, raw_data_format["X"]],
                agent_track[i, raw_data_format["Y"]],
            )

            # Compute distances for all the social tracks
            for social_track in social_tracks[:, i, :]:

                neigh_x = social_track[raw_data_format["X"]]
                neigh_y = social_track[raw_data_format["Y"]]
                if viz:
                    plt.scatter(neigh_x, neigh_y, color="green")

                # Distance between agent and social
                instant_distance = np.sqrt((agent_x - neigh_x)**2 +
                                           (agent_y - neigh_y)**2)

                # If not a neighbor, continue
                if instant_distance > self.NEARBY_DISTANCE_THRESHOLD:
                    continue

                # Check if the social track is in front or back
                is_front_or_back = self.get_is_front_or_back(
                    agent_track[:2, :] if i == 0 else agent_track[:i + 1, :],
                    neigh_x,
                    neigh_y,
                    raw_data_format,
                )
                if is_front_or_back == "front":
                    min_distance_front_and_back[i, 0] = min(
                        min_distance_front_and_back[i, 0], instant_distance)

                elif is_front_or_back == "back":
                    min_distance_front_and_back[i, 1] = min(
                        min_distance_front_and_back[i, 1], instant_distance)

            if viz:
                plt.scatter(agent_x, agent_y, color="red")
                plt.text(
                    agent_track[i, raw_data_format["X"]],
                    agent_track[i, raw_data_format["Y"]],
                    "{0:.1f}".format(min_distance_front_and_back[i, 0]),
                    fontsize=5,
                )

        if viz:
            plt.text(
                agent_track[0, raw_data_format["X"]],
                agent_track[0, raw_data_format["Y"]],
                "s",
                fontsize=12,
            )
            plt.text(
                agent_track[-1, raw_data_format["X"]],
                agent_track[-1, raw_data_format["Y"]],
                "e",
                fontsize=12,
            )
            plt.axis("equal")
            plt.show()
        return min_distance_front_and_back

    def get_num_neighbors(
            self,
            agent_track: np.ndarray,
            social_tracks: np.ndarray,
            obs_len: int,
            raw_data_format: Dict[str, int],
    ) -> np.ndarray:
        """Get minimum distance of the tracks in front and back.

        Args:
            agent_track (numpy array): Data for the agent track
            social_tracks (numpy array): Array of relevant tracks
            obs_len (int): Length of observed trajectory
            raw_data_format (Dict): Format of the sequence
        Returns:
            num_neighbors (numpy array): Number of neighbors at each timestep

        """
        num_neighbors = np.full((obs_len, 1), 0)

        for i in range(obs_len):

            agent_x, agent_y = (
                agent_track[i, raw_data_format["X"]],
                agent_track[i, raw_data_format["Y"]],
            )

            for social_track in social_tracks[:, i, :]:

                neigh_x = social_track[raw_data_format["X"]]
                neigh_y = social_track[raw_data_format["Y"]]

                instant_distance = np.sqrt((agent_x - neigh_x)**2 +
                                           (agent_y - neigh_y)**2)

                if instant_distance < self.NEARBY_DISTANCE_THRESHOLD:
                    num_neighbors[i, 0] += 1

        return num_neighbors

    def compute_social_features(
            self,
            df: pd.DataFrame,
            agent_track: np.ndarray,
            obs_len: int,
            seq_len: int,
            raw_data_format: Dict[str, int],
    ) -> np.ndarray:
        """Compute social features for the given sequence.

        Social features are meant to capture social context. 
        Here we use minimum distance to the vehicle in front, to the vehicle in back, 
        and number of neighbors as social features.

        Args:
            df (pandas Dataframe): Dataframe containing all the tracks in the sequence
            agent_track (numpy array): Data for the agent track
            obs_len (int): Length of observed trajectory
            seq_len (int): Length of the sequence
            raw_data_format (Dict): Format of the sequence
        Returns:
            social_features (numpy array): Social features for the agent track

        """
        agent_ts = np.sort(np.unique(df["TIMESTAMP"].values))

        if agent_ts.shape[0] == obs_len:
            df_obs = df
            agent_track_obs = agent_track

        else:
            # Get obs dataframe and agent track
            df_obs = df[df["TIMESTAMP"] < agent_ts[obs_len]]
            assert (np.unique(df_obs["TIMESTAMP"].values).shape[0] == obs_len
                    ), "Obs len mismatch"
            agent_track_obs = agent_track[:obs_len]

        # Filter out non-relevant tracks
        social_tracks_obs = self.filter_tracks(df_obs, obs_len,
                                               raw_data_format)

        # Get minimum following distance in front and back
        min_distance_front_and_back_obs = self.get_min_distance_front_and_back(
            agent_track_obs,
            social_tracks_obs,
            obs_len,
            raw_data_format,
            viz=False)

        # Get number of neighbors
        num_neighbors_obs = self.get_num_neighbors(agent_track_obs,
                                                   social_tracks_obs, obs_len,
                                                   raw_data_format)

        # Agent track with social features
        social_features_obs = np.concatenate(
            (min_distance_front_and_back_obs, num_neighbors_obs), axis=1)
        social_features = np.full((seq_len, social_features_obs.shape[1]),
                                  None)
        social_features[:obs_len] = social_features_obs

        return social_features
