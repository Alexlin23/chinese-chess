"""Gate SGD updates on newly committed original self-play positions."""


class UpdateSchedule:
    def __init__(self, interval, total_positions, saved=None):
        if type(interval) is not int or interval < 0:
            raise ValueError("new_positions_per_update must be a nonnegative integer")
        self.interval = interval
        # First enablement / a changed interval starts counting at this resume.
        self.consumed_positions = total_positions
        if interval and saved and saved.get("interval") == interval:
            cursor = saved.get("consumed_positions")
            if type(cursor) is not int or not 0 <= cursor <= total_positions:
                raise ValueError("update schedule/replay position counter mismatch")
            self.consumed_positions = cursor

    def ready(self, total_positions):
        return not self.interval or total_positions - self.consumed_positions >= self.interval

    def record_update(self, total_positions):
        if not self.ready(total_positions):
            raise ValueError("not enough new replay positions for an SGD update")
        if self.interval:
            self.consumed_positions += self.interval

    def state_dict(self):
        return dict(interval=self.interval, consumed_positions=self.consumed_positions)
