from enum import Enum, Flag, auto
from typing import TypedDict

import scipy.spatial.transform as st

S_TO_NS = 1_000_000_000


class ObsType(Flag):
    RGB = auto()
    COLOR_VISUAL = RGB

    LANGUAGE = auto()
    LANGUAGE_EMBEDDING = auto()

    BINARY_GRIPPER = auto()

    CARTESIAN_POS = auto()
    JOINT_POS = auto()
    EUCLIDIAN = CARTESIAN_POS | JOINT_POS

    ROTATION_MATRIX = auto()
    QUAT = auto()
    SPHERICAL = ROTATION_MATRIX | QUAT

    POSE_MATRIX = auto()

    # there exists a sensible way to interpolate between any two.
    # specifically, it is a connected smooth Riemannian manifold.
    INTERPOLABLE = EUCLIDIAN | SPHERICAL | POSE_MATRIX

    PROPER_SUBSET = ROTATION_MATRIX | QUAT | POSE_MATRIX
    PERSISTENT = LANGUAGE | LANGUAGE_EMBEDDING

    def get_category_name(self) -> str:
        if self in ObsType.COLOR_VISUAL:
            return "visual"

        if self in ObsType.LANGUAGE:
            return "language"

        if self in ObsType.EUCLIDIAN:
            return "euclidian"

        if self in ObsType.SPHERICAL:
            return "spherical"

        if self in ObsType.POSE_MATRIX:
            return "pose_matrix"

        msg = f"Unknown category for {self}"
        raise ValueError(msg)


class LieRepr(Flag):
    ABSOLUTE = auto()
    RELATIVE = auto()

    INDEPENDENTLY_INTERPOLABLE = ABSOLUTE | RELATIVE

    # there are some cases where interpolating a value that is in principle interpolable requires knowing other values.
    # say you are commanding deltas to the current end-effector pose that the controller converts to a target pose
    # that it then zero-holds until a new command arrives. the correct interpolation would compute this target pose
    # based on the proprio at the timestamp of the last delta, zero-hold it, and convert it back to a delta based on
    # the interpolated proprio. you can't zero-hold the delta because you would be too fast and might overshoot,
    # and you can't "shrink" the delta directly due to tracking error you might have had during teleop.
    # this control mode happens in robosuite 1.4 which is behind libero. this case is not implemented.
    # there's usually no need to interpolate sim data bc (unlike in real!) all the data comes in at the same constant
    # frequency and you're gonna need the actions at that frequency too.
    OTHER = auto()


class NormalizationType(Enum):
    NONE = "skip"  # don't create a normalizer for this key.
    IDENTITY = "none"  # create a normalizer for this key but do nothing (ugly way of supporting concatting keys where some should be normalized and some not).
    SQUASH = "squash"  # the one from TRI, 2nd and 98th percentiles become -1 and 1, clamping at -1.5 and 1.5.
    SQUASH_HARD = "squash_hard"  # same as SQUASH but afterwards (post-clamp) rescale to [-1, 1]. You need this e.g. if you want to FAST tokenize after.
    VARIANCE = "variance"  # clamp at the same boundary as SQUASH and normalize such that afterwards dataset variance is 1 and mean 0.


class ObsMeta(TypedDict):
    obs_type: ObsType
    shift_right_by: float  # seconds
    repr: LieRepr | None
    horizon: int
    target_frequency: int  # Hz
    normalization_type: NormalizationType
    target_repr: LieRepr


SCIPY_ROTATION_CONVERSIONS = {
    ObsType.ROTATION_MATRIX: (st.Rotation.from_matrix, st.Rotation.as_matrix),
    ObsType.QUAT: (st.Rotation.from_quat, st.Rotation.as_quat),
}
