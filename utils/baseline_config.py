"""This module defines all the config parameters."""

FEATURE_FORMAT = {
    "TIMESTAMP": 0,
    "TRACK_ID": 1,
    "OBJECT_TYPE": 2,
    "X": 3,
    "Y": 4,
    "CITY_NAME": 5,
    "MIN_DISTANCE_FRONT": 6,
    "MIN_DISTANCE_BACK": 7,
    "NUM_NEIGHBORS": 8,
    "OFFSET_FROM_CENTERLINE": 9,
    "DISTANCE_ALONG_CENTERLINE": 10,
}

RAW_DATA_FORMAT = {
    "TIMESTAMP": 0,
    "TRACK_ID": 1,
    "OBJECT_TYPE": 2,
    "X": 3,
    "Y": 4,
    "CITY_NAME": 5,
}

LSTM_HELPER_DICT_IDX = {
    "CENTROIDS": 0,
    "CITY_NAMES": 1,
    "CANDIDATE_CENTERLINES": 2,
    "CANDIDATE_NT_DISTANCES": 3,
    "TRANSLATION": 4,
    "ROTATION": 5,
    "CANDIDATE_DELTA_REFERENCES": 6,
    "DELTA_REFERENCE": 7,
    "SEQ_PATHS": 8,
}

BASELINE_INPUT_FEATURES = {
    "social":
    ["X", "Y", "MIN_DISTANCE_FRONT", "MIN_DISTANCE_BACK", "NUM_NEIGHBORS"],
    "map": ["OFFSET_FROM_CENTERLINE", "DISTANCE_ALONG_CENTERLINE"],
    "map_social": [
        "OFFSET_FROM_CENTERLINE",
        "DISTANCE_ALONG_CENTERLINE",
        "MIN_DISTANCE_FRONT",
        "MIN_DISTANCE_BACK",
        "NUM_NEIGHBORS",
    ],
    "none": ["X", "Y"],
}

BASELINE_OUTPUT_FEATURES = {
    "social": ["X", "Y"],
    "map": ["OFFSET_FROM_CENTERLINE", "DISTANCE_ALONG_CENTERLINE"],
    "map_social": ["OFFSET_FROM_CENTERLINE", "DISTANCE_ALONG_CENTERLINE"],
    "none": ["X", "Y"],
}

# Feature computation
_FEATURES_SMALL_SIZE = 100

# Map Feature computations
_MANHATTAN_THRESHOLD = 5.0  # meters
_DFS_THRESHOLD_FRONT_SCALE = 45.0  # meters
_DFS_THRESHOLD_BACK_SCALE = 40.0  # meters
_MAX_SEARCH_RADIUS_CENTERLINES = 50.0  # meters
_MAX_CENTERLINE_CANDIDATES_TEST = 10

# Social Feature computation
PADDING_TYPE = "REPEAT"  # Padding type for partial sequences
STATIONARY_THRESHOLD = (
    13)  # index of the sorted velocity to look at, to call it as stationary
VELOCITY_THRESHOLD = 1.0  # Velocity threshold for stationary
EXIST_THRESHOLD = (
    15
)  # Number of timesteps the track should exist to be considered in social context
DEFAULT_MIN_DIST_FRONT_AND_BACK = 100.0  # Default front/back distance
NEARBY_DISTANCE_THRESHOLD = 50.0  # Distance threshold to call a track as neighbor
FRONT_OR_BACK_OFFSET_THRESHOLD = 5.0  # Offset threshold from direction of travel
