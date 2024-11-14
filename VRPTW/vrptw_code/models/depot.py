class Depot:
    def __init__(self, x, y, ready_time, due_date):
        self.x = x
        self.y = y
        self.ready_time = ready_time
        self.due_date = due_date
        self.service_time = 0  # starting service_time with 0 value