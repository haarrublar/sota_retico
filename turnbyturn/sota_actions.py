import threading
import time

from sota_thinclient.pose import Command, LedID, ServoID, SotaState


class SotaActions:
    """All of Sota's physical actions (LEDs, movements) in one place."""

    def __init__(
        self,
        sota,
        speaker_module=None,
        led_on="#FF0000",
        led_off="#000000",
        off_delay=1.0,
        start_timeout=10.0,
    ):
        self.sota = sota
        self.speaker = speaker_module  # used to know when Sota stops talking
        self.led_on = led_on
        self.led_off = led_off
        self.off_delay = off_delay  # extra time after the speaker finishes
        self.start_timeout = start_timeout  # give up if Sota never starts talking
        self._watcher = None
        self.sota.pose.enable()

    # ---------- LEDs ----------
    def set_leds(self, leds: dict, msec=50):
        self._send(SotaState(leds=leds), msec)

    def mouth_on(self, *_):
        self.set_leds({LedID.MOUTH: self.led_on})
        if self.speaker and (self._watcher is None or not self._watcher.is_alive()):
            self._watcher = threading.Thread(target=self._auto_mouth_off, daemon=True)
            self._watcher.start()

    def mouth_off(self, *_):
        self.set_leds({LedID.MOUTH: self.led_off})

    def eyes(self, color):
        self.set_leds({LedID.LEFT_EYE: color, LedID.RIGHT_EYE: color})

    def _auto_mouth_off(self):
        start = time.time()
        while self.speaker.busy_until <= time.time():  # wait for Sota to start
            if time.time() - start > self.start_timeout:
                break
            time.sleep(0.05)
        while self.speaker.busy_until > time.time():  # wait for Sota to finish
            time.sleep(0.05)
        time.sleep(self.off_delay)
        self.mouth_off()

    # ---------- movements ----------
    def move(self, joints: dict, msec=500):
        """joints: {ServoID.HEAD_YAW: 0.3, ...} in radians."""
        self._send(SotaState(joint_space=joints), msec)

    # ---------- internals ----------
    def _send(self, state, msec, command=Command.APPEND):
        try:
            self.sota.pose.send_command(state, msec=msec, command=command)
        except Exception as e:
            print("Sota action error:", e)

    def shutdown(self):
        self.mouth_off()
        self.sota.pose.disable()
