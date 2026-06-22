from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import builtin_interfaces.msg

from perception_dataset.rosbag2.rosbag2_reader import Rosbag2Reader
from perception_dataset.utils.rosbag2 import stamp_to_unix_timestamp

if TYPE_CHECKING:
    from autoware_vehicle_msgs.msg import (
        GearReport,
        HazardLightsReport,
        SteeringReport,
        TurnIndicatorsReport,
        VelocityReport,
    )
    from tier4_vehicle_msgs.msg import ActuationStatusStamped, SteeringWheelStatusStamped


class VehicleStatusHandler:
    """A class to handle ROS2 messages related to vehicle status."""

    # Field name[str] -> Topic name[str]
    DEFAULT_TOPIC_MAPPING = {
        "actuation_status": "/vehicle/status/actuation_status",
        "control_mode": "/vehicle/status/control_mode",
        "door_status": "/vehicle/status/door_status",
        "gear_status": "/vehicle/status/gear_status",
        "hazard_lights_status": "/vehicle/status/hazard_lights_status",
        "steering_status": "/vehicle/status/steering_status",
        "steering_wheel_status": "/vehicle/status/steering_wheel_status",
        "turn_indicators_status": "/vehicle/status/turn_indicators_status",
        "velocity_status": "/vehicle/status/velocity_status",
    }

    # GearReport[uint8] -> Shift[str]
    GEAR2SHIFT: Dict[int, str] = {
        0: "NONE",
        1: "NEUTRAL",
        2: "DRIVE",
        3: "DRIVE",
        4: "DRIVE",
        5: "DRIVE",
        6: "DRIVE",
        7: "DRIVE",
        8: "DRIVE",
        9: "DRIVE",
        10: "DRIVE",
        11: "DRIVE",
        12: "DRIVE",
        13: "DRIVE",
        14: "DRIVE",
        15: "DRIVE",
        16: "DRIVE",
        17: "DRIVE",
        18: "DRIVE",
        19: "DRIVE",
        20: "REVERSE",
        21: "REVERSE",
        22: "PARK",
        23: "LOW",
        24: "LOW",
    }

    # Turn-signal state derived from TurnIndicatorsReport.report (autoware_vehicle_msgs).
    # Constants: DISABLE=1, ENABLE_LEFT=2, ENABLE_RIGHT=3. There is no 0 and no hazard value -
    # hazard is reported separately by HazardLightsReport (see HAZARD_MAPPING). Each entry is the
    # {left, right} state; the hazard field is filled in by `build_indicators`.
    INDICATOR_MAPPING: Dict[int, Dict[str, str]] = {
        1: {"left": "off", "right": "off"},  # DISABLE
        2: {"left": "on", "right": "off"},  # ENABLE_LEFT
        3: {"left": "off", "right": "on"},  # ENABLE_RIGHT
    }

    # Hazard state derived from HazardLightsReport.report (autoware_vehicle_msgs).
    # Constants: DISABLE=1, ENABLE=2.
    HAZARD_MAPPING: Dict[int, str] = {
        1: "off",  # DISABLE
        2: "on",  # ENABLE
    }

    def __init__(self, bag_dir: str, *, topic_mapping: Optional[Dict[str, str]] = None) -> None:
        """Construct a new object.

        Args:
            bag_dir (str): Directory path of rosbag.
            topic_mapping (Optional[Dict[str, str]], optional): Mapping of a field name to a topic name.
                If `None`, `DEFAULT_TOPIC_MAPPING` will be used. Defaults to None.
        """
        self._reader = Rosbag2Reader(bag_dir=bag_dir)
        self._topic_mapping = self.get_topic_mapping(topic_mapping=topic_mapping)

        buffer = {
            key: [msg for msg in self._reader.read_messages(topics=[topic])]  # noqa
            for key, topic in self._topic_mapping.items()
        }
        self._buffer = buffer

    @classmethod
    def get_topic_mapping(cls, topic_mapping: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Return a mapping of a field name to a topic name.
        If `topic_mapping` is not specified, `DEFAULT_TOPIC_MAPPING` will be returned.

        Args:
            topic_mapping (Optional[Dict[str, str]], optional): Custom mapping. Defaults to None.

        Returns:
            Dict[str, str]: Topic mapping.
        """
        if topic_mapping is not None:
            assert set(cls.DEFAULT_TOPIC_MAPPING) == set(topic_mapping.keys())
            return topic_mapping
        return cls.DEFAULT_TOPIC_MAPPING

    def gear_to_shift(self, gear: int) -> str:
        """Convert the value of the gear report to the shift state using `GEAR2SHIFT`.

        Args:
            gear (int): Value of `GearReport.report`.

        Returns:
            str: Shift state.
        """
        return self.GEAR2SHIFT[gear]

    @classmethod
    def build_indicators(
        cls, turn_report: int, hazard_report: Optional[int] = None
    ) -> Dict[str, str]:
        """Build the T4 ``indicators`` mapping ``{left, right, hazard}``.

        ``left`` / ``right`` come from ``TurnIndicatorsReport.report`` (``turn_report``); ``hazard``
        comes from ``HazardLightsReport.report`` (``hazard_report``), which is published on a
        separate topic. ``hazard`` defaults to ``"off"`` when no hazard report is provided. Unknown
        report values fall back to the off state (no exception) so a single malformed message cannot
        abort the conversion.

        Args:
            turn_report (int): Value of ``TurnIndicatorsReport.report``.
            hazard_report (Optional[int]): Value of ``HazardLightsReport.report``, or ``None`` to
                leave ``hazard`` off.

        Returns:
            Dict[str, str]: Mapping with keys [left, right, hazard], values "on"/"off".
        """
        state = dict(cls.INDICATOR_MAPPING.get(turn_report, {"left": "off", "right": "off"}))
        state["hazard"] = (
            cls.HAZARD_MAPPING.get(hazard_report, "off") if hazard_report is not None else "off"
        )
        return state

    def indicator_to_state(self, indicator: int) -> Dict[str, str]:
        """Convert a `TurnIndicatorsReport.report` value into the `{left, right, hazard}` mapping.

        Kept for backward compatibility; `hazard` is always `"off"` here because it cannot be
        derived from the turn-indicators message alone. Use `build_indicators` to also populate
        `hazard` from a `HazardLightsReport.report` value.

        Args:
            indicator (int): Value of `TurnIndicatorsReport.report`.

        Returns:
            Dict[str, str]: Mapping value for each indicator as following format:
                - keys: [left, right, hazard]
                - values: "on" or "off"
        """
        return self.build_indicators(indicator)

    def get_closest_msg(self, key: str, stamp: builtin_interfaces.msg.Time) -> Any:
        assert key in self._buffer
        messages: List[Any] = self._buffer[key]
        timestamp = stamp_to_unix_timestamp(stamp)

        def time_diff(x) -> float:
            if hasattr(x, "header"):
                return abs(stamp_to_unix_timestamp(x.header.stamp) - timestamp)
            elif hasattr(x, "stamp"):
                return abs(stamp_to_unix_timestamp(x.stamp) - timestamp)
            else:
                raise ValueError("Unexpected message type")

        return min(messages, key=lambda x: time_diff(x))

    def get_actuation_statuses(self) -> List[ActuationStatusStamped]:
        """Return a list of `ActuationStatusStamped`s.

        Returns:
            List[ActuationStatusStamped]: List of messages.
        """
        return self._buffer["actuation_status"]

    def get_gear_reports(self) -> List[GearReport]:
        """Return a list of `GearReport`s.

        Returns:
            List[GearReport]: List of messages.
        """
        return self._buffer["gear_status"]

    def get_steering_reports(self) -> List[SteeringReport]:
        """Return a list of `SteeringReport`s.

        Returns:
            List[SteeringReport]: List of messages.
        """
        return self._buffer["steering_status"]

    def get_steering_wheel_statuses(self) -> List[SteeringWheelStatusStamped]:
        """Return a list of `SteeringWheelStatusStamped`s.

        Returns:
            List[SteeringWheelStatusStamped]: List of messages.
        """
        return self._buffer["steering_wheel_status"]

    def get_turn_indicators_reports(self) -> List[TurnIndicatorsReport]:
        """Return a list of `TurnIndicatorsReport`s.

        Returns:
            List[TurnIndicatorsReport]: List of messages.
        """
        return self._buffer["turn_indicators_status"]

    def get_hazard_lights_reports(self) -> List[HazardLightsReport]:
        """Return a list of `HazardLightsReport`s

        Returns:
            List[HazardLightsReport]: List of messages.
        """
        return self._buffer["hazard_lights_status"]

    def get_velocity_reports(self) -> List[VelocityReport]:
        """Return a list of `VelocityReport`s.

        Returns:
            List[VelocityReport]: List of messages.
        """
        return self._buffer["velocity_status"]
