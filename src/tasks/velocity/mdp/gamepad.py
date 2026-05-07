"""Non-blocking Linux joystick reader for Xbox controllers.

Reads from /dev/input/jsN using the Linux joystick API. No external
dependencies required.

Axis mapping (Xbox Wireless Controller):
  0: Left stick X  (left=-32767, right=+32767)
  1: Left stick Y  (up=-32767, down=+32767)
  2: Left trigger   (released=-32767, pressed=+32767)
  3: Right stick X
  4: Right stick Y
  5: Right trigger
  6: D-pad X
  7: D-pad Y
"""

from __future__ import annotations

import os
import struct
import threading
from typing import Final

# Linux joystick event format: uint32 time, int16 value, uint8 type, uint8 number
_JS_EVENT_FMT: Final = "IhBB"
_JS_EVENT_SIZE: Final = struct.calcsize(_JS_EVENT_FMT)

# Event types.
_JS_EVENT_BUTTON: Final = 0x01
_JS_EVENT_AXIS: Final = 0x02
_JS_EVENT_INIT: Final = 0x80


class Gamepad:
    """Non-blocking gamepad reader that runs a background thread."""

    def __init__(self, device: str = "/dev/input/js0", deadzone: float = 0.1) -> None:
        self.device = device
        self.deadzone = deadzone

        self._axes: dict[int, float] = {}  # axis_number -> normalized [-1, 1]
        self._buttons: dict[int, bool] = {}  # button_number -> pressed
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self) -> bool:
        """Start the background reader. Returns True if device was opened."""
        if not os.path.exists(self.device):
            print(f"[Gamepad] Device not found: {self.device}")
            return False
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def axis(self, number: int) -> float:
        """Get normalized axis value in [-1, 1] with deadzone applied."""
        with self._lock:
            raw = self._axes.get(number, 0.0)
        if abs(raw) < self.deadzone:
            return 0.0
        return raw

    def button(self, number: int) -> bool:
        with self._lock:
            return self._buttons.get(number, False)

    # Xbox stick helpers (normalized, deadzone-applied).

    @property
    def left_x(self) -> float:
        """Left stick horizontal: left=-1, right=+1."""
        return self.axis(0)

    @property
    def left_y(self) -> float:
        """Left stick vertical: up=+1, down=-1 (inverted from raw)."""
        return -self.axis(1)

    @property
    def right_x(self) -> float:
        return self.axis(2)

    @property
    def right_y(self) -> float:
        return -self.axis(3)

    def _read_loop(self) -> None:
        try:
            fd = os.open(self.device, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as e:
            print(f"[Gamepad] Cannot open {self.device}: {e}")
            return

        self._connected = True
        print(f"[Gamepad] Connected: {self.device}")

        while self._running:
            try:
                data = os.read(fd, _JS_EVENT_SIZE)
                if len(data) != _JS_EVENT_SIZE:
                    continue
            except BlockingIOError:
                # No data available — sleep briefly to avoid busy-wait.
                import time
                time.sleep(0.005)
                continue
            except OSError:
                break

            _time, value, etype, number = struct.unpack(_JS_EVENT_FMT, data)
            etype &= ~_JS_EVENT_INIT  # strip init flag

            with self._lock:
                if etype == _JS_EVENT_AXIS:
                    self._axes[number] = value / 32767.0
                elif etype == _JS_EVENT_BUTTON:
                    self._buttons[number] = bool(value)

        os.close(fd)
        self._connected = False
        print("[Gamepad] Disconnected")
