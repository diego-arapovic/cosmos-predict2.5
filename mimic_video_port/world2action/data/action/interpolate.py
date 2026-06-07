import numpy as np
import scipy.spatial.transform as st

from .types import SCIPY_ROTATION_CONVERSIONS, ObsMeta, ObsType


def get_closest_indices(actual_timestamps: np.ndarray, requested_timestamps: np.ndarray) -> np.ndarray:
    return np.abs(actual_timestamps[None] - requested_timestamps[:, None]).argmin(axis=1)


def choose_closest(values: np.ndarray, actual_timestamps: np.ndarray, requested_timestamps: np.ndarray) -> np.ndarray:
    return values[get_closest_indices(actual_timestamps, requested_timestamps)]


def get_previous_indices(actual_timestamps: np.ndarray, requested_timestamps: np.ndarray) -> np.ndarray:
    return np.maximum(np.searchsorted(actual_timestamps, requested_timestamps, side="right") - 1, 0)


def choose_previous(values: np.ndarray, actual_timestamps: np.ndarray, requested_timestamps: np.ndarray):
    return values[get_previous_indices(actual_timestamps, requested_timestamps)]


def interpolate_1d(values: np.ndarray, value_indices: np.ndarray, interpolate_idxs: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(interpolate_idxs, value_indices, column) for column in values.T], axis=0).T


def slerp(values: st.Rotation, value_indices: np.ndarray, interpolate_idxs: np.ndarray) -> st.Rotation:
    return st.Slerp(times=value_indices, rotations=values)(interpolate_idxs)


def interpolate_pose_matrix(values: np.ndarray, value_indices: np.ndarray, interpolate_idxs: np.ndarray) -> np.ndarray:
    rots = st.Rotation.from_matrix(values[:, :3, :3])
    interpolated_rots = slerp(rots, value_indices, interpolate_idxs).as_matrix()

    trans = values[:, :3, 3]
    interpolated_trans = interpolate_1d(trans, value_indices, interpolate_idxs)

    num_steps = len(interpolate_idxs)
    res = np.tile(np.eye(4, dtype=np.float32), (num_steps, 1)).reshape(num_steps, 4, 4)
    res[:, :3, :3] = interpolated_rots
    res[:, :3, 3] = interpolated_trans
    return res


def interpolate_lowdim_non_delta(
    values: np.ndarray,
    value_indices: np.ndarray,
    interpolate_idxs: np.ndarray,
    obs_type: ObsType,
) -> np.ndarray:
    if obs_type in ObsType.EUCLIDIAN:
        return interpolate_1d(values, value_indices, interpolate_idxs)

    if obs_type in ObsType.SPHERICAL:
        from_rot, to_rot = SCIPY_ROTATION_CONVERSIONS[obs_type]
        return to_rot(slerp(from_rot(values), value_indices, interpolate_idxs))

    if obs_type == ObsType.POSE_MATRIX:
        return interpolate_pose_matrix(values, value_indices, interpolate_idxs)

    raise NotImplementedError


def interpolate_lowdim(
    values: np.ndarray,
    actual_timestamps: np.ndarray,
    requested_timestamps: np.ndarray,
    meta: ObsMeta,
    *,
    is_action: bool,
) -> np.ndarray:
    interpolate_idxs = np.clip(requested_timestamps, actual_timestamps[0], actual_timestamps[-1])
    return interpolate_lowdim_non_delta(values, actual_timestamps, interpolate_idxs, meta["obs_type"])
