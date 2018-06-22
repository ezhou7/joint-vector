import time


class Timer:
    def __init__(self):
        self.active_timers = dict()

    def start(self, timer_name):
        self.active_timers[timer_name] = time.time()

    def end(self, timer_name):
        time_elapsed = self.active_timers[timer_name] - time.time()
        del self.active_timers[timer_name]
        return time_elapsed


stopwatch = Timer()
