import numpy as np

class Customer:
    def __init__(self, cust_no, x_coord, y_coord, demand, ready_time, due_date, service_time):
        self.cust_no = cust_no
        self.x = x_coord
        self.y = y_coord
        self.demand = demand
        self.ready_time = ready_time
        self.due_date = due_date
        self.service_time = service_time

class ProblemInstance:
    def __init__(self, filename):
        self.filename = filename
        self.depot = None
        self.customers = []
        self.vehicle_capacity = None
        self.num_customers = 0
        self.distance_matrix = None
        self.parse_problem()
        self.calculate_distance_matrix()

    def parse_problem(self):
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        # removing the empty lines and strip whitespace
        lines = [line.strip() for line in lines if line.strip() != '']

        # reading of vehicle capacity
        capacity_line_index = lines.index('NUMBER     CAPACITY') + 1
        capacity_line = lines[capacity_line_index].split()
        self.vehicle_capacity = int(capacity_line[-1])

        # finding index where customer data starts
        customer_header = 'CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE   TIME'
        customer_header_index = lines.index(customer_header) + 1

        # reading customer data
        for line in lines[customer_header_index:]:
            tokens = line.strip().split()
            if len(tokens) < 7:
                continue
            cust_no = int(tokens[0])
            x_coord = float(tokens[1])
            y_coord = float(tokens[2])
            demand = float(tokens[3])
            ready_time = float(tokens[4])
            due_date = float(tokens[5])
            service_time = float(tokens[6])

            customer = Customer(cust_no, x_coord, y_coord, demand, ready_time, due_date, service_time)

            if cust_no == 0:
                self.depot = customer
            else:
                self.customers.append(customer)

        self.num_customers = len(self.customers)

    def calculate_distance_matrix(self):
        num_nodes = self.num_customers + 1  # remember there is a depot
        self.distance_matrix = np.zeros((num_nodes, num_nodes))
        nodes = [self.depot] + self.customers
        for i in range(num_nodes):
            for j in range(num_nodes):
                dx = nodes[i].x - nodes[j].x
                dy = nodes[i].y - nodes[j].y
                self.distance_matrix[i][j] = np.sqrt(dx * dx + dy * dy)
