"""Unit tests for the turn-indicator / hazard-light state mapping in VehicleStatusHandler.

These cover the report-value -> {left, right, hazard} conversion only, which is pure data and needs
no real ROS / rosbag2 environment (the handler imports the Autoware overlay message types under
TYPE_CHECKING only, so the module imports without a sourced overlay).

Message constants (autoware_vehicle_msgs):
  TurnIndicatorsReport.report:  DISABLE=1, ENABLE_LEFT=2, ENABLE_RIGHT=3
  HazardLightsReport.report:    DISABLE=1, ENABLE=2
"""

import pytest

from perception_dataset.ros2.vehicle_msgs.vehicle_status_handler import VehicleStatusHandler

DISABLE = 1
ENABLE_LEFT = 2
ENABLE_RIGHT = 3

HAZARD_DISABLE = 1
HAZARD_ENABLE = 2


class TestBuildIndicators:
    @pytest.mark.parametrize(
        "turn_report, expected_left, expected_right",
        [
            (DISABLE, "off", "off"),
            (ENABLE_LEFT, "on", "off"),
            (ENABLE_RIGHT, "off", "on"),
        ],
    )
    def test_left_right_from_turn_report(self, turn_report, expected_left, expected_right):
        state = VehicleStatusHandler.build_indicators(turn_report)
        assert state["left"] == expected_left
        assert state["right"] == expected_right
        # hazard defaults to off when no hazard report is supplied.
        assert state["hazard"] == "off"

    @pytest.mark.parametrize(
        "hazard_report, expected_hazard",
        [
            (None, "off"),
            (HAZARD_DISABLE, "off"),
            (HAZARD_ENABLE, "on"),
        ],
    )
    def test_hazard_from_hazard_report(self, hazard_report, expected_hazard):
        state = VehicleStatusHandler.build_indicators(DISABLE, hazard_report)
        assert state["hazard"] == expected_hazard

    def test_left_and_hazard_combined(self):
        state = VehicleStatusHandler.build_indicators(ENABLE_LEFT, HAZARD_ENABLE)
        assert state == {"left": "on", "right": "off", "hazard": "on"}

    def test_always_has_all_three_keys(self):
        state = VehicleStatusHandler.build_indicators(ENABLE_RIGHT, HAZARD_ENABLE)
        assert set(state.keys()) == {"left", "right", "hazard"}

    def test_unknown_turn_report_falls_back_to_off(self):
        # An unexpected report value must not raise (a single malformed message can't abort a run).
        state = VehicleStatusHandler.build_indicators(99)
        assert state == {"left": "off", "right": "off", "hazard": "off"}

    def test_unknown_hazard_report_falls_back_to_off(self):
        state = VehicleStatusHandler.build_indicators(ENABLE_LEFT, 99)
        assert state == {"left": "on", "right": "off", "hazard": "off"}


class TestIndicatorToState:
    """`indicator_to_state` is the backward-compatible turn-only wrapper (hazard always off)."""

    @pytest.mark.parametrize(
        "turn_report, expected",
        [
            (DISABLE, {"left": "off", "right": "off", "hazard": "off"}),
            (ENABLE_LEFT, {"left": "on", "right": "off", "hazard": "off"}),
            (ENABLE_RIGHT, {"left": "off", "right": "on", "hazard": "off"}),
        ],
    )
    def test_returns_corrected_left_right_with_hazard_off(self, turn_report, expected):
        handler = VehicleStatusHandler.__new__(VehicleStatusHandler)  # no rosbag2 needed
        assert handler.indicator_to_state(turn_report) == expected


class TestMappingTables:
    """Regression guards against the previous off-by-one mapping (0/1/2/3 with hazard-in-turn)."""

    def test_indicator_mapping_is_keyed_by_real_constants(self):
        assert set(VehicleStatusHandler.INDICATOR_MAPPING.keys()) == {
            DISABLE,
            ENABLE_LEFT,
            ENABLE_RIGHT,
        }
        # The turn-indicator mapping must NOT carry a hazard key (hazard is a separate topic).
        for state in VehicleStatusHandler.INDICATOR_MAPPING.values():
            assert set(state.keys()) == {"left", "right"}

    def test_hazard_mapping_values(self):
        assert VehicleStatusHandler.HAZARD_MAPPING == {HAZARD_DISABLE: "off", HAZARD_ENABLE: "on"}

    def test_gear2shift_values_are_valid_shiftstates(self):
        # Every GEAR2SHIFT value must be a member of t4_devkit's ShiftState enum, else writing
        # vehicle_state raises ValueError (the old "DRIVE" was not a ShiftState; valid is "FORWARD").
        from t4_devkit.schema.tables.vehicle_state import ShiftState

        valid = {s.value for s in ShiftState}
        for gear, shift in VehicleStatusHandler.GEAR2SHIFT.items():
            assert shift in valid, f"GEAR2SHIFT[{gear}]={shift!r} is not a valid ShiftState"
