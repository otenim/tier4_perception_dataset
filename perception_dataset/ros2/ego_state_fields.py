"""Topic-agnostic sources for the optional ego_pose fields (twist / acceleration / geocoordinate).

Each source reads one freely chosen topic, time-interpolates the value to a query stamp, and returns
it for ego_pose.json. Values are taken from the message as-is (no sign heuristics); twist and
acceleration are rotated into the ego child frame via /tf_static only when the source message frame
differs from it. geocoordinate (NavSatFix) is stored verbatim with no frame transform.

This replaces the former INSHandler, which hard-coded INS topics and applied an ad-hoc orientation
/ sign correction. The module avoids importing ROS message types or Rosbag2Reader at runtime (it
duck-types messages and the reader), so the extraction/interpolation logic is unit-testable without
a sourced ROS environment.

NOTE: out-of-range query stamps currently extrapolate (twist/acceleration silently, geocoordinate
with a warning), preserving the previous behavior. Replacing extrapolation with null-on-out-of-range
is deliberately left to a follow-up change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple
import warnings

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

from perception_dataset.utils.logger import configure_logger

if TYPE_CHECKING:
    from perception_dataset.rosbag2.rosbag2_reader import Rosbag2Reader

logger = configure_logger(modname=__name__)

# sensor_msgs/NavSatStatus.STATUS_NO_FIX (inlined to keep this module import-light).
_NAVSAT_STATUS_NO_FIX = -1

TWIST_SUPPORTED_TYPES = (
    "nav_msgs/Odometry",
    "geometry_msgs/TwistStamped",
    "geometry_msgs/TwistWithCovarianceStamped",
)
ACCELERATION_SUPPORTED_TYPES = (
    "sensor_msgs/Imu",
    "geometry_msgs/AccelStamped",
    "geometry_msgs/AccelWithCovarianceStamped",
)
GEOCOORDINATE_SUPPORTED_TYPES = ("sensor_msgs/NavSatFix",)


def _stamp_to_unix(stamp) -> float:
    """Duck-typed ROS Time (sec/nanosec) -> unix seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _msg_name(type_str: Optional[str]) -> str:
    """ "pkg/msg/Type" or "pkg/Type" -> "Type"."""
    return type_str.split("/")[-1] if type_str else ""


# --- per-type extractors: msg -> (vector(s), source_frame_id) -------------------------------------
# Each returns the value expressed in `source_frame_id` (the frame it is later rotated FROM).
def _twist_from_odometry(msg) -> Tuple[Tuple[float, ...], Tuple[float, ...], str]:
    t = msg.twist.twist  # Odometry.twist is a TwistWithCovariance; .twist is the Twist
    return (
        (t.linear.x, t.linear.y, t.linear.z),
        (t.angular.x, t.angular.y, t.angular.z),
        msg.child_frame_id,  # REP 105: Odometry twist is in child_frame_id
    )


def _twist_from_twist_stamped(msg) -> Tuple[Tuple[float, ...], Tuple[float, ...], str]:
    t = msg.twist
    return (
        (t.linear.x, t.linear.y, t.linear.z),
        (t.angular.x, t.angular.y, t.angular.z),
        msg.header.frame_id,
    )


def _twist_from_twist_cov_stamped(msg) -> Tuple[Tuple[float, ...], Tuple[float, ...], str]:
    t = msg.twist.twist
    return (
        (t.linear.x, t.linear.y, t.linear.z),
        (t.angular.x, t.angular.y, t.angular.z),
        msg.header.frame_id,
    )


_TWIST_EXTRACTORS: dict[str, Callable] = {
    "Odometry": _twist_from_odometry,
    "TwistStamped": _twist_from_twist_stamped,
    "TwistWithCovarianceStamped": _twist_from_twist_cov_stamped,
}


def _accel_from_imu(msg) -> Tuple[Tuple[float, ...], str]:
    a = msg.linear_acceleration
    return ((a.x, a.y, a.z), msg.header.frame_id)


def _accel_from_accel_stamped(msg) -> Tuple[Tuple[float, ...], str]:
    a = msg.accel.linear
    return ((a.x, a.y, a.z), msg.header.frame_id)


def _accel_from_accel_cov_stamped(msg) -> Tuple[Tuple[float, ...], str]:
    a = msg.accel.accel.linear
    return ((a.x, a.y, a.z), msg.header.frame_id)


_ACCEL_EXTRACTORS: dict[str, Callable] = {
    "Imu": _accel_from_imu,
    "AccelStamped": _accel_from_accel_stamped,
    "AccelWithCovarianceStamped": _accel_from_accel_cov_stamped,
}


class _VectorInterpolator:
    """Linear, per-component interpolation of a fixed-length vector sampled over time.

    Out-of-range queries extrapolate (interp1d fill_value="extrapolate"), preserving the previous
    INS behavior; optionally a single warning is emitted on the first out-of-range query.
    """

    def __init__(
        self,
        timestamps: Sequence[float],
        vectors: Sequence[Sequence[float]],
        *,
        warn_out_of_range: bool = False,
        name: str = "",
    ) -> None:
        t = np.asarray(timestamps, dtype=float)
        v = np.asarray(vectors, dtype=float)
        if t.shape[0] == 0:
            raise ValueError(f"No samples to interpolate for {name!r}.")
        self._name = name
        self._warn_out_of_range = warn_out_of_range
        self._warned = False
        if t.shape[0] == 1:
            # interp1d needs >=2 points; a single sample is a constant.
            self._single: Optional[np.ndarray] = v[0]
            self._funcs: Optional[List[Callable]] = None
            self._tmin = self._tmax = float(t[0])
        else:
            self._single = None
            self._funcs = [
                interp1d(t, v[:, i], fill_value="extrapolate") for i in range(v.shape[1])
            ]
            self._tmin, self._tmax = float(t.min()), float(t.max())

    def __call__(self, stamp_unix: float) -> np.ndarray:
        if (
            self._warn_out_of_range
            and not self._warned
            and (stamp_unix < self._tmin or stamp_unix > self._tmax)
        ):
            warnings.warn(
                f"Query stamp is outside the observed range of {self._name!r}; "
                "the value is extrapolated and may be inaccurate."
            )
            self._warned = True
        if self._single is not None:
            return self._single.copy()
        return np.array([float(f(stamp_unix)) for f in self._funcs], dtype=float)


def _resolve_rotation(
    reader: "Rosbag2Reader",
    child_frame_id: str,
    source_frame_id: str,
    sample_stamp,
    *,
    context: str,
) -> Optional[Rotation]:
    """Static rotation mapping a vector from source_frame_id into child_frame_id, or None when no
    rotation is needed (frames equal/empty). Raises with a clear message if the static transform is
    unavailable (e.g. the frame is not connected in /tf_static)."""
    if not source_frame_id or source_frame_id == child_frame_id:
        return None
    try:
        tf = reader.get_transform_stamped(
            target_frame=child_frame_id,
            source_frame=source_frame_id,
            stamp=sample_stamp,
        )
    except Exception as e:  # tf2 raises various lookup/extrapolation errors
        raise ValueError(
            f"{context}: cannot rotate from source frame '{source_frame_id}' into ego child frame "
            f"'{child_frame_id}'. A static transform between them must exist in /tf_static. ({e})"
        ) from e
    q = tf.transform.rotation
    return Rotation.from_quat([q.x, q.y, q.z, q.w])


class TwistSource:
    """twist (vx, vy, vz, yaw_rate, pitch_rate, roll_rate) in the ego child frame."""

    def __init__(self, reader: "Rosbag2Reader", topic: str, child_frame_id: str) -> None:
        type_str = reader.get_topic_type(topic)
        if type_str is None:
            raise ValueError(f"twist_topic '{topic}' is not present in the rosbag.")
        extractor = _TWIST_EXTRACTORS.get(_msg_name(type_str))
        if extractor is None:
            raise ValueError(
                f"twist_topic '{topic}' has unsupported message type '{type_str}'. "
                f"Supported: {', '.join(TWIST_SUPPORTED_TYPES)}."
            )
        timestamps: List[float] = []
        linear: List[Tuple[float, ...]] = []
        angular: List[Tuple[float, ...]] = []
        source_frame: Optional[str] = None
        first_stamp = None
        for msg in reader.read_messages(topics=[topic]):
            lin, ang, frame = extractor(msg)
            timestamps.append(_stamp_to_unix(msg.header.stamp))
            linear.append(lin)
            angular.append(ang)
            if source_frame is None:
                source_frame = frame
                first_stamp = msg.header.stamp
        if not timestamps:
            raise ValueError(f"twist_topic '{topic}' contains no messages.")
        self._linear = _VectorInterpolator(timestamps, linear, name=f"{topic} (linear)")
        self._angular = _VectorInterpolator(timestamps, angular, name=f"{topic} (angular)")
        self._rotation = _resolve_rotation(
            reader, child_frame_id, source_frame, first_stamp, context=f"twist_topic '{topic}'"
        )

    def lookup(self, stamp) -> Tuple[float, float, float, float, float, float]:
        ts = _stamp_to_unix(stamp)
        lin = self._linear(ts)
        ang = self._angular(ts)
        if self._rotation is not None:
            lin = self._rotation.apply(lin)
            ang = self._rotation.apply(ang)
        # Schema order: (vx, vy, vz, yaw_rate, pitch_rate, roll_rate) = (lin, wz, wy, wx).
        return (
            float(lin[0]),
            float(lin[1]),
            float(lin[2]),
            float(ang[2]),
            float(ang[1]),
            float(ang[0]),
        )


class AccelerationSource:
    """acceleration (ax, ay, az) in the ego child frame."""

    def __init__(self, reader: "Rosbag2Reader", topic: str, child_frame_id: str) -> None:
        type_str = reader.get_topic_type(topic)
        if type_str is None:
            raise ValueError(f"acceleration_topic '{topic}' is not present in the rosbag.")
        extractor = _ACCEL_EXTRACTORS.get(_msg_name(type_str))
        if extractor is None:
            raise ValueError(
                f"acceleration_topic '{topic}' has unsupported message type '{type_str}'. "
                f"Supported: {', '.join(ACCELERATION_SUPPORTED_TYPES)}."
            )
        timestamps: List[float] = []
        values: List[Tuple[float, ...]] = []
        source_frame: Optional[str] = None
        first_stamp = None
        for msg in reader.read_messages(topics=[topic]):
            val, frame = extractor(msg)
            timestamps.append(_stamp_to_unix(msg.header.stamp))
            values.append(val)
            if source_frame is None:
                source_frame = frame
                first_stamp = msg.header.stamp
        if not timestamps:
            raise ValueError(f"acceleration_topic '{topic}' contains no messages.")
        self._interp = _VectorInterpolator(timestamps, values, name=f"{topic}")
        self._rotation = _resolve_rotation(
            reader,
            child_frame_id,
            source_frame,
            first_stamp,
            context=f"acceleration_topic '{topic}'",
        )

    def lookup(self, stamp) -> Tuple[float, float, float]:
        acc = self._interp(_stamp_to_unix(stamp))
        if self._rotation is not None:
            acc = self._rotation.apply(acc)
        return (float(acc[0]), float(acc[1]), float(acc[2]))


class GeoCoordinateSource:
    """geocoordinate (latitude, longitude, altitude) from NavSatFix, stored verbatim."""

    def __init__(self, reader: "Rosbag2Reader", topic: str) -> None:
        type_str = reader.get_topic_type(topic)
        if type_str is None:
            raise ValueError(f"geocoordinate_topic '{topic}' is not present in the rosbag.")
        if _msg_name(type_str) != "NavSatFix":
            raise ValueError(
                f"geocoordinate_topic '{topic}' has unsupported message type '{type_str}'. "
                f"Supported: {', '.join(GEOCOORDINATE_SUPPORTED_TYPES)}."
            )
        timestamps: List[float] = []
        coords: List[Tuple[float, ...]] = []
        for msg in reader.read_messages(topics=[topic]):
            if msg.status.status == _NAVSAT_STATUS_NO_FIX:
                continue
            timestamps.append(_stamp_to_unix(msg.header.stamp))
            coords.append((msg.latitude, msg.longitude, msg.altitude))
        if not timestamps:
            raise ValueError(
                f"geocoordinate_topic '{topic}' contains no valid NavSatFix messages "
                "(empty or all NO_FIX)."
            )
        self._interp = _VectorInterpolator(
            timestamps, coords, warn_out_of_range=True, name=f"{topic}"
        )

    def lookup(self, stamp) -> Tuple[float, float, float]:
        coord = self._interp(_stamp_to_unix(stamp))
        return (float(coord[0]), float(coord[1]), float(coord[2]))
