"""Regression tests for converter table initialization.

The annotated converter (`task: convert_rosbag2_to_t4`) overrides `_init_tables()` with a
deliberately reduced set of tables, but it still inherits `_convert()` from the base
`_Rosbag2ToNonAnnotatedT4Converter`. That base `_convert()` calls `_convert_vehicle_state`
when `with_vehicle_status` is set, and that writes into `self._vehicle_state_table`. If the
override forgets to create that table, the annotated path crashes with
`AttributeError: ... has no attribute '_vehicle_state_table'`.

This test pins the override to keep `_vehicle_state_table` so `with_vehicle_status` works on
the annotated converter.
"""

from perception_dataset.rosbag2.rosbag2_to_t4_converter import _Rosbag2ToT4Converter


def test_annotated_converter_init_tables_creates_vehicle_state_table():
    # `_init_tables()` only reads `self._sensor_enums`; build the instance via `__new__` to skip
    # the bag-reading `__init__` so the test stays fast and free of a rosbag fixture.
    converter = _Rosbag2ToT4Converter.__new__(_Rosbag2ToT4Converter)
    converter._sensor_enums = []

    converter._init_tables()

    assert hasattr(converter, "_vehicle_state_table"), (
        "_Rosbag2ToT4Converter._init_tables() must create _vehicle_state_table so the "
        "with_vehicle_status path inherited from the base converter does not crash on the "
        "annotated (convert_rosbag2_to_t4) path."
    )
