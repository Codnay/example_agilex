"""Quest reader thread - reads controller data and manages teleop state."""

from __future__ import annotations

import os
import time
import traceback

import numpy as np

from common.configs import (
    CONTROLLER_DATA_RATE,
    GRIP_THRESHOLD,
    ROTATION_SCALE,
    TRANSLATION_SCALE,
)
from common.data_manager import DataManager, RobotActivityState
from meta_quest_teleop.reader import MetaQuestReader


def _call_first_successful(
    obj: object,
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]],
) -> object | None:
    for name, args, kwargs in calls:
        method = getattr(obj, name, None)
        if not callable(method):
            continue
        try:
            return method(*args, **kwargs)
        except TypeError:
            continue
    return None


def _get_right_grip_value(quest_reader: MetaQuestReader) -> float:
    value = _call_first_successful(
        quest_reader,
        [
            ("get_grip_value", ("right",), {}),
            ("get_grip_value", (), {"hand": "right"}),
            ("getGripValue", ("right",), {}),
            ("getGripValue", (), {"hand": "right"}),
        ],
    )
    try:
        if value is None:
            return 0.0
        if isinstance(value, (tuple, list, np.ndarray)):
            if len(value) == 0:
                return 0.0
            return float(value[0])
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _get_right_trigger_value(quest_reader: MetaQuestReader) -> float:
    value = _call_first_successful(
        quest_reader,
        [
            ("get_trigger_value", ("right",), {}),
            ("get_trigger_value", (), {"hand": "right"}),
            ("getTriggerValue", ("right",), {}),
            ("getTriggerValue", (), {"hand": "right"}),
        ],
    )
    try:
        if value is None:
            return 0.0
        if isinstance(value, (tuple, list, np.ndarray)):
            if len(value) == 0:
                return 0.0
            return float(value[0])
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _get_right_hand_transform_ros(quest_reader: MetaQuestReader) -> np.ndarray | None:
    transform = _call_first_successful(
        quest_reader,
        [
            ("get_hand_controller_transform_ros", (), {"hand": "right"}),
            ("get_hand_controller_transform_ros", ("right",), {}),
            ("get_hand_transform_ros", (), {"hand": "right"}),
            ("get_hand_transform_ros", ("right",), {}),
        ],
    )
    if transform is None:
        return None
    try:
        arr = np.asarray(transform, dtype=np.float64)
        return arr.copy() if arr.shape == (4, 4) else None
    except Exception:
        return None


def quest_reader_thread(
    data_manager: DataManager, quest_reader: MetaQuestReader
) -> None:
    """Quest reader thread - reads controller data and manages teleop state.

    This thread runs at high frequency to ensure responsive controller input.
    Handles:
    - Reading Meta Quest controller data
    - Processing grip button (dead man's switch)
    - Managing teleop activation/deactivation
    - Capturing initial poses when teleop activates

    Args:
        data_manager: DataManager object for thread-safe communication
        quest_reader: MetaQuestReader instance
    """
    print("🎮 Quest Controller thread started")

    dt: float = 1.0 / CONTROLLER_DATA_RATE
    prev_grip_active: bool = False
    printed_api_warning: bool = False
    debug_inputs: bool = os.getenv("QUEST_DEBUG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    last_debug_print_time: float = 0.0
    last_seen_transform_keys: str = ""

    try:
        while not data_manager.is_shutdown_requested():
            iteration_start = time.time()

            # Get controller data (be tolerant to MetaQuestReader API differences)
            try:
                grip_value = _get_right_grip_value(quest_reader)
                trigger_value = _get_right_trigger_value(quest_reader)
                controller_transform_raw = _get_right_hand_transform_ros(quest_reader)
            except Exception as read_err:
                if not printed_api_warning:
                    printed_api_warning = True
                    print(
                        "⚠️  Quest reader polling failed; continuing with defaults. "
                        f"Error: {read_err}"
                    )
                grip_value = 0.0
                trigger_value = 0.0
                controller_transform_raw = None

            # Update shared state with controller data
            data_manager.set_controller_data(
                controller_transform_raw, grip_value, trigger_value
            )

            controller_transform_filt, _, _ = data_manager.get_controller_data()

            # Grip button logic (dead man's switch)
            robot_activity_state = data_manager.get_robot_activity_state()
            # Teleop can only be activated if robot is ENABLED (not HOMING or DISABLED)
            grip_active = (
                grip_value >= GRIP_THRESHOLD
                and robot_activity_state == RobotActivityState.ENABLED
            )

            # Rising edge - grip just pressed AND robot is enabled
            if (
                grip_active
                and not prev_grip_active
                and controller_transform_filt is not None
            ):
                # Start teleop control
                # capture initial poses
                controller_initial_transform = controller_transform_filt.copy()

                # Capture initial robot end-effector pose
                robot_initial_transform = data_manager.get_current_end_effector_pose()

                data_manager.set_teleop_state(
                    True, controller_initial_transform, robot_initial_transform
                )

                print("✓ Teleop control activated")
                print(
                    f"  Controller initial position: {controller_initial_transform[:3, 3]}"
                )
                if robot_initial_transform is not None:
                    print(f"  Robot initial position: {robot_initial_transform[:3, 3]}")
                else:
                    print("  Robot initial position: None")

            # Falling edge - grip just released OR robot disabled
            elif not grip_active and prev_grip_active:
                # Stop teleop control
                data_manager.set_teleop_state(False, None, None)
                print("✗ Teleop control deactivated")

            prev_grip_active = grip_active

            if debug_inputs and (iteration_start - last_debug_print_time) >= 1.0:
                last_debug_print_time = iteration_start
                teleop_active = data_manager.get_teleop_active()
                current_joint_angles = data_manager.get_current_joint_angles()
                current_ee = data_manager.get_current_end_effector_pose()

                # Attempt to read any non-controller transform as "headset" for debugging.
                headset_pos_str = "None"
                try:
                    transforms, _buttons = quest_reader.get_transformations_and_buttons()
                    keys = sorted(list(transforms.keys())) if transforms else []
                    last_seen_transform_keys = ",".join(keys) if keys else ""

                    headset_key = None
                    for candidate in ("h", "head", "H", "Head", "headset", "0"):
                        if transforms and candidate in transforms:
                            headset_key = candidate
                            break
                    if headset_key is None and transforms:
                        for k in keys:
                            if k not in {"r", "l"}:
                                headset_key = k
                                break

                    if headset_key is not None and transforms is not None:
                        t = np.asarray(transforms[headset_key], dtype=np.float64)
                        if t.shape == (4, 4):
                            p = t[:3, 3]
                            headset_pos_str = (
                                f"{headset_key}=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})"
                            )
                        else:
                            headset_pos_str = f"{headset_key}=(bad_shape)"
                except Exception:
                    # Keep debug output robust; headset pose is optional.
                    pass

                raw_pos_str = "None"
                filt_pos_str = "None"
                if controller_transform_raw is not None:
                    p = controller_transform_raw[:3, 3]
                    raw_pos_str = f"({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})"
                if controller_transform_filt is not None:
                    p = controller_transform_filt[:3, 3]
                    filt_pos_str = f"({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})"

                delta_pos_str = "None"
                delta_rot_str = "None"
                if teleop_active and controller_transform_filt is not None:
                    controller_initial, robot_initial = (
                        data_manager.get_initial_robot_controller_transforms()
                    )
                    if controller_initial is not None:
                        dp = controller_transform_filt[:3, 3] - controller_initial[:3, 3]
                        delta_pos_str = (
                            f"({dp[0]:.3f},{dp[1]:.3f},{dp[2]:.3f})|norm={float(np.linalg.norm(dp)):.3f}"
                        )
                        # rotation delta angle (deg)
                        dR = controller_transform_filt[:3, :3] @ controller_initial[:3, :3].T
                        tr = float(np.trace(dR))
                        cos_angle = (tr - 1.0) / 2.0
                        cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
                        angle_deg = float(np.degrees(np.arccos(cos_angle)))
                        delta_rot_str = f"{angle_deg:.1f}deg|scaled={angle_deg * ROTATION_SCALE:.1f}"

                ee_pos_str = "None"
                if current_ee is not None and current_ee.shape == (4, 4):
                    p = current_ee[:3, 3]
                    ee_pos_str = f"({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})"

                joints_str = (
                    "None"
                    if current_joint_angles is None
                    else "[" + ",".join(f"{x:.1f}" for x in current_joint_angles) + "]"
                )
                print(
                    "🎮 Quest input debug | "
                    f"robot_state={robot_activity_state.value} | "
                    f"grip={grip_value:.3f} (threshold={GRIP_THRESHOLD}) | "
                    f"trigger={trigger_value:.3f} | "
                    f"teleop_active={teleop_active} | "
                    f"transform_raw={'ok' if controller_transform_raw is not None else 'None'} | "
                    f"transform_filt={'ok' if controller_transform_filt is not None else 'None'} | "
                    f"headset_pos={headset_pos_str} | "
                    f"ctrl_pos_raw={raw_pos_str} | ctrl_pos_filt={filt_pos_str} | "
                    f"delta_pos={delta_pos_str} (scale={TRANSLATION_SCALE}) | "
                    f"delta_rot={delta_rot_str} (scale={ROTATION_SCALE}) | "
                    f"ee_pos={ee_pos_str} | "
                    f"joints_deg={joints_str}"
                )
                if last_seen_transform_keys:
                    print(f"🎮 Quest input debug | transform_keys={last_seen_transform_keys}")

            # Sleep to maintain loop rate (check shutdown more frequently)
            elapsed = time.time() - iteration_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"❌ Quest reader thread error: {e}")
        traceback.print_exc()
        data_manager.request_shutdown()
    finally:
        # Ensure clean exit - deactivate teleop
        data_manager.set_teleop_state(False, None, None)
        print("🎮 Quest Controller thread stopped")
