"""Unit tests for perception_dataset.ros2.ego_state_fields.

These avoid a real ROS / rosbag2 environment by duck-typing messages and the reader with
SimpleNamespace, mirroring how the source classes access them.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from perception_dataset.ros2.ego_state_fields import (
    AccelerationSource,
    GeoCoordinateSource,
    TwistSource,
    _msg_name,
    _VectorInterpolator,
)


def _stamp(sec: float):
    whole = int(sec)
    return SimpleNamespace(sec=whole, nanosec=int(round((sec - whole) * 1e9)))


def _vec(x, y, z):
    return SimpleNamespace(x=float(x), y=float(y), z=float(z))


def _quat_xyzw(rot: Rotation):
    x, y, z, w = rot.as_quat()
    return SimpleNamespace(x=x, y=y, z=z, w=w)


class FakeReader:
    """Minimal stand-in for Rosbag2Reader."""

    def __init__(self, types, messages, transforms=None):
        self._types = types  # topic -> type string
        self._messages = messages  # topic -> [msg, ...]
        self._transforms = transforms or {}  # (target, source) -> Rotation

    def get_topic_type(self, topic):
        return self._types.get(topic)

    def read_messages(self, topics):
        for msg in self._messages.get(topics[0], []):
            yield msg

    def get_transform_stamped(self, target_frame, source_frame, stamp):
        rot = self._transforms.get((target_frame, source_frame))
        if rot is None:
            raise LookupError(f"no static transform {target_frame} <- {source_frame}")
        return SimpleNamespace(transform=SimpleNamespace(rotation=_quat_xyzw(rot)))


# --- message builders -----------------------------------------------------------------------------
def _odometry(t, lin, ang, child_frame_id="base_link"):
    return SimpleNamespace(
        header=SimpleNamespace(stamp=_stamp(t), frame_id="map"),
        child_frame_id=child_frame_id,
        twist=SimpleNamespace(twist=SimpleNamespace(linear=_vec(*lin), angular=_vec(*ang))),
    )


def _twist_cov_stamped(t, lin, ang, frame_id):
    return SimpleNamespace(
        header=SimpleNamespace(stamp=_stamp(t), frame_id=frame_id),
        twist=SimpleNamespace(twist=SimpleNamespace(linear=_vec(*lin), angular=_vec(*ang))),
    )


def _imu(t, acc, frame_id="base_link"):
    return SimpleNamespace(
        header=SimpleNamespace(stamp=_stamp(t), frame_id=frame_id),
        linear_acceleration=_vec(*acc),
    )


def _accel_cov_stamped(t, acc, frame_id):
    return SimpleNamespace(
        header=SimpleNamespace(stamp=_stamp(t), frame_id=frame_id),
        accel=SimpleNamespace(accel=SimpleNamespace(linear=_vec(*acc), angular=_vec(0, 0, 0))),
    )


def _navsatfix(t, lat, lon, alt, status=0):
    return SimpleNamespace(
        header=SimpleNamespace(stamp=_stamp(t), frame_id="gnss"),
        status=SimpleNamespace(status=status),
        latitude=lat,
        longitude=lon,
        altitude=alt,
    )


# --- helpers --------------------------------------------------------------------------------------
def test_msg_name_strips_package():
    assert _msg_name("nav_msgs/msg/Odometry") == "Odometry"
    assert _msg_name("geometry_msgs/AccelWithCovarianceStamped") == "AccelWithCovarianceStamped"
    assert _msg_name(None) == ""


def test_vector_interpolator_linear_midpoint():
    interp = _VectorInterpolator([0.0, 2.0], [(0.0, 10.0), (2.0, 30.0)])
    np.testing.assert_allclose(interp(1.0), [1.0, 20.0])


def test_vector_interpolator_single_sample_is_constant():
    interp = _VectorInterpolator([5.0], [(7.0, 8.0)])
    np.testing.assert_allclose(interp(0.0), [7.0, 8.0])
    np.testing.assert_allclose(interp(100.0), [7.0, 8.0])


def test_vector_interpolator_extrapolates_out_of_range():
    # Out-of-range currently extrapolates (preserved behavior), not clamped/null.
    interp = _VectorInterpolator([0.0, 1.0], [(0.0,), (1.0,)])
    np.testing.assert_allclose(interp(3.0), [3.0])


# --- TwistSource ----------------------------------------------------------------------------------
def test_twist_source_odometry_schema_order_no_rotation():
    msgs = [
        _odometry(0.0, lin=(1.0, 2.0, 3.0), ang=(0.1, 0.2, 0.3)),
        _odometry(2.0, lin=(3.0, 4.0, 5.0), ang=(0.3, 0.4, 0.5)),
    ]
    reader = FakeReader({"/odom": "nav_msgs/msg/Odometry"}, {"/odom": msgs})
    src = TwistSource(
        reader, "/odom", child_frame_id="base_link"
    )  # source frame == child -> no rot
    out = src.lookup(_stamp(1.0))  # midpoint
    # (vx, vy, vz, yaw_rate=wz, pitch_rate=wy, roll_rate=wx)
    np.testing.assert_allclose(out, [2.0, 3.0, 4.0, 0.4, 0.3, 0.2])


def test_twist_source_rotates_into_child_frame():
    # Static transform base_link <- sensor is a +90 deg yaw; a sensor-frame (1,0,0) -> (0,1,0).
    rot = Rotation.from_euler("z", 90, degrees=True)
    msgs = [
        _twist_cov_stamped(0.0, lin=(1.0, 0.0, 0.0), ang=(0.0, 0.0, 1.0), frame_id="sensor"),
        _twist_cov_stamped(2.0, lin=(1.0, 0.0, 0.0), ang=(0.0, 0.0, 1.0), frame_id="sensor"),
    ]
    reader = FakeReader(
        {"/t": "geometry_msgs/msg/TwistWithCovarianceStamped"},
        {"/t": msgs},
        transforms={("base_link", "sensor"): rot},
    )
    src = TwistSource(reader, "/t", child_frame_id="base_link")
    vx, vy, vz, yaw_rate, pitch_rate, roll_rate = src.lookup(_stamp(1.0))
    np.testing.assert_allclose([vx, vy, vz], [0.0, 1.0, 0.0], atol=1e-9)
    # angular (0,0,1) rotated by +90deg yaw stays (0,0,1); yaw_rate is the z-component.
    np.testing.assert_allclose([yaw_rate, pitch_rate, roll_rate], [1.0, 0.0, 0.0], atol=1e-9)


def test_twist_source_unsupported_type_raises():
    reader = FakeReader({"/t": "std_msgs/msg/Float64"}, {"/t": []})
    with pytest.raises(ValueError, match="unsupported message type"):
        TwistSource(reader, "/t", child_frame_id="base_link")


def test_twist_source_missing_static_transform_raises():
    msgs = [_twist_cov_stamped(0.0, (1, 0, 0), (0, 0, 0), frame_id="sensor")]
    reader = FakeReader(
        {"/t": "geometry_msgs/msg/TwistWithCovarianceStamped"}, {"/t": msgs}, transforms={}
    )
    with pytest.raises(ValueError, match="static transform"):
        TwistSource(reader, "/t", child_frame_id="base_link")


# --- AccelerationSource ---------------------------------------------------------------------------
def test_acceleration_source_imu():
    msgs = [_imu(0.0, (1.0, 2.0, 9.8)), _imu(2.0, (3.0, 4.0, 9.8))]
    reader = FakeReader({"/imu": "sensor_msgs/msg/Imu"}, {"/imu": msgs})
    src = AccelerationSource(reader, "/imu", child_frame_id="base_link")
    np.testing.assert_allclose(src.lookup(_stamp(1.0)), [2.0, 3.0, 9.8])


def test_acceleration_source_accel_with_covariance_stamped():
    msgs = [
        _accel_cov_stamped(0.0, (0.0, 0.0, 0.0), frame_id="base_link"),
        _accel_cov_stamped(2.0, (2.0, 2.0, 2.0), frame_id="base_link"),
    ]
    reader = FakeReader({"/acc": "geometry_msgs/msg/AccelWithCovarianceStamped"}, {"/acc": msgs})
    src = AccelerationSource(reader, "/acc", child_frame_id="base_link")
    np.testing.assert_allclose(src.lookup(_stamp(1.0)), [1.0, 1.0, 1.0])


def test_acceleration_source_missing_topic_raises():
    reader = FakeReader({}, {})
    with pytest.raises(ValueError, match="not present in the rosbag"):
        AccelerationSource(reader, "/nope", child_frame_id="base_link")


# --- GeoCoordinateSource --------------------------------------------------------------------------
def test_geocoordinate_source_interpolates_and_skips_no_fix():
    msgs = [
        _navsatfix(0.0, 35.0, 139.0, 10.0, status=0),
        _navsatfix(1.0, 0.0, 0.0, 0.0, status=-1),  # NO_FIX -> skipped
        _navsatfix(2.0, 35.2, 139.2, 30.0, status=0),
    ]
    reader = FakeReader({"/fix": "sensor_msgs/msg/NavSatFix"}, {"/fix": msgs})
    src = GeoCoordinateSource(reader, "/fix")
    np.testing.assert_allclose(src.lookup(_stamp(1.0)), [35.1, 139.1, 20.0])


def test_geocoordinate_source_unsupported_type_raises():
    reader = FakeReader({"/fix": "sensor_msgs/msg/Imu"}, {"/fix": []})
    with pytest.raises(ValueError, match="unsupported message type"):
        GeoCoordinateSource(reader, "/fix")
