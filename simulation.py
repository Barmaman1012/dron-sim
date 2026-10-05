class Simulation:
    def __init__(self, dt: float, duration: float):
        self.dt = dt
        self.duration = duration
        self.step_count = 0

    @property
    def time(self):
        return self.step_count * self.dt

    def step(self):
        self.step_count += 1

    def run(self):
        while self.time < self.duration:
            self.step()