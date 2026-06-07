import numpy as np

from .types import LieRepr, ObsType


def pose_matrix_absolute_to_relative(abs_poses: np.ndarray, base_pose: np.ndarray) -> np.ndarray:
    return np.linalg.inv(base_pose) @ abs_poses


def convert_to_repr(
    value: np.ndarray,
    obs_type: ObsType,
    original_repr: LieRepr,
    target_repr: LieRepr,
    *,
    is_action: bool | None = None,
    relative_base_value: np.ndarray | None = None,
) -> np.ndarray:
    if original_repr == target_repr:
        return value

    if target_repr == LieRepr.ABSOLUTE:
        msg = "Cannot convert from a non-absolute representation to an absolute one. The information isn't there."
        raise ValueError(msg)

    obs_type_category_name = obs_type.get_category_name()
    kwargs = {"obs_type": obs_type} if obs_type in ObsType.SPHERICAL else {}

    if is_action is relative_base_value:
        msg = "Exactly one of `is_action` or `relative_base_value` must be non-None."
        raise ValueError(msg)

    if is_action is not None:
        relative_base_value = value[0 if is_action else -1]

    fn_name = f"{obs_type_category_name}_absolute_to_relative"
    if fn_name not in globals():
        msg = (
            "You need to extend the conversion file for your case as it was"
            " not required for the mimic-video bridge and libero experiments."
        )
        raise NotImplementedError(msg)

    return globals()[fn_name](value, relative_base_value, **kwargs)
