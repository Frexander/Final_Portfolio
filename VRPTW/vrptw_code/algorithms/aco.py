import numpy as np
import random
from models.problem_instance import ProblemInstance
from models.vehicle import Vehicle

class AntColonyOptimization:
    def __init__(
        self,
        problem_instance,
        num_ants,
        num_iterations,
        alpha,
        beta,
        evaporation_rate,
        Q,
        penalty_time=1000,
        penalty_capacity=1000
    ):
        """
        initialize for Ant Colony Optimization algorithm

        take following paramters
        - problem_instance (ProblemInstance): VRPTW problem instance
        - num_ants (int): number of ants in the colony
        - num_iterations (int): number of iterations to run the algorithm
        - alpha (float): pheromone importance factor
        - beta (float): heuristic importance factor
        - evaporation_rate (float): the rate at which pheromones evaporate
        - Q (float): pheromone deposit factor
        - penalty_time (float): penalty coefficient for time window violations
        - penalty_capacity (float): penalty coefficient for capacity violations
        """
        self.problem = problem_instance
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha  # pheromone importance
        self.beta = beta    # heuristic importance
        self.evaporation_rate = evaporation_rate
        self.Q = Q  # pheromone deposit factor
        self.penalty_time = penalty_time
        self.penalty_capacity = penalty_capacity
        self.pheromone_matrix = np.ones(
            (self.problem.num_customers + 1, self.problem.num_customers + 1)
        )
        self.best_distance = float('inf')
        self.best_solution = None

    def run(self):
        """
        running the Ant Colony Optimization algorithm

        returns back
        - best_solution (list): the best set of routes found
        - best_distance (float): The total distance of the best solution
        - best_distance_history (list): the tracking of best distances per iteration
        """
        best_distance_history = []
        for iteration in range(self.num_iterations):
            all_routes = []
            all_distances = []
            for ant in range(self.num_ants):
                routes, total_distance = self.construct_solution()
                all_routes.append(routes)
                all_distances.append(total_distance)
                if total_distance < self.best_distance:
                    self.best_distance = total_distance
                    self.best_solution = routes
            # appending best distance to history
            best_distance_history.append(self.best_distance)
            self.update_pheromones(all_routes, all_distances)
            print(
                f"Iteration {iteration+1}/{self.num_iterations}, "
                f"Best Distance: {self.best_distance:.2f}"
            )
        return self.best_solution, self.best_distance, best_distance_history

    def construct_solution(self):
        """
        construct a solution for each ant by probabilistically choosing the next customer

        returns back
        - routes: List of routes constructed by the ant
        - total_distance (float): total distance of the constructed routes
        """
        unvisited = set(range(1, self.problem.num_customers + 1))
        routes = []
        total_distance = 0.0
        penalty = 0.0
        while unvisited:
            vehicle = Vehicle(self.problem.vehicle_capacity)
            vehicle.current_location = 0  # depot
            vehicle.current_time = self.problem.depot.ready_time
            vehicle.load = 0
            route = [0]  # depot
            while True:
                feasible_customers = []
                for cust_no in unvisited:
                    customer = self.problem.customers[cust_no - 1]
                    if vehicle.load + customer.demand <= vehicle.capacity:
                        travel_time = self.problem.distance_matrix[
                            vehicle.current_location
                        ][cust_no]
                        arrival_time = vehicle.current_time + travel_time
                        waiting_time = max(0, customer.ready_time - arrival_time)
                        service_begin_time = arrival_time + waiting_time
                        if service_begin_time <= customer.due_date:
                            feasible_customers.append(cust_no)
                if not feasible_customers:
                    break
                probabilities = self.calculate_probabilities(
                    vehicle.current_location, feasible_customers
                )
                next_customer = self.select_next_customer(probabilities)
                unvisited.remove(next_customer)
                customer = self.problem.customers[next_customer - 1]
                # calculating times
                travel_time = self.problem.distance_matrix[
                    vehicle.current_location
                ][next_customer]
                arrival_time = vehicle.current_time + travel_time
                waiting_time = max(0, customer.ready_time - arrival_time)
                service_begin_time = arrival_time + waiting_time
                # updating the state of vehicle
                vehicle.current_time = service_begin_time + customer.service_time
                vehicle.load += customer.demand
                vehicle.current_location = next_customer
                route.append(next_customer)
                # adding the distance
                total_distance += travel_time
            # back to depot
            travel_time_to_depot = self.problem.distance_matrix[
                vehicle.current_location
            ][0]
            arrival_time_at_depot = vehicle.current_time + travel_time_to_depot
            total_distance += travel_time_to_depot
            route.append(0)
            # cheaking depot time window
            if arrival_time_at_depot > self.problem.depot.due_date:
                penalty += self.penalty_time * (
                    arrival_time_at_depot - self.problem.depot.due_date
                )
            routes.append(route)
        total_distance += penalty
        return routes, total_distance

    def calculate_probabilities(self, current_location, feasible_customers):
        """
        calculate the probabilities of moving to each feasible customer

        take following paramters
        - current_location (int): The current node index
        - feasible_customers: List of feasible customer indices

        returns back
        - probabilities : List of tuples (customer_index, probability)
        """
        pheromones = []
        heuristic = []
        for cust_no in feasible_customers:
            tau = self.pheromone_matrix[current_location][cust_no] ** self.alpha
            # calculates the influence of pheromones: tau = pheromone value raised in alpha
            eta = (
                1.0 / self.problem.distance_matrix[current_location][cust_no]
            ) ** self.beta
            #calculates heuristic information: eta = (1 / distance) raised in beta
            pheromones.append(tau)
            heuristic.append(eta)
        pheromones = np.array(pheromones)
        heuristic = np.array(heuristic)
        probabilities = pheromones * heuristic
        # combines pheromone and heuristic information
        probabilities /= probabilities.sum()
        # normalizes the probabilities so that they sum to 1
        return list(zip(feasible_customers, probabilities))

    def select_next_customer(self, probabilities):
        """
        selecting the next customer based on calculated probabilities

        take following paramters
        - probabilities: List of tuples (customer_index, probability)

        returns back
        - next_customer (int): the selected next customer index.
        """
        rnd = random.random()
        cumulative = 0.0
        for cust_no, prob in probabilities:
            cumulative += prob
            if rnd <= cumulative:
                return cust_no
        return probabilities[-1][0]  # for when rounding errors apppears

    def update_pheromones(self, all_routes, all_distances):
        """
        updating the pheromone matrix based on the routes and distances

        take following paramters
        - all_routes : list of routes from all ants
        - all_distances (list): corresponding distances of the routes
        """
        # evaporation
        self.pheromone_matrix *= (1 - self.evaporation_rate)
        # the pheromones evaporate by a factor of (1 - the evaporation rate)
        for routes, distance in zip(all_routes, all_distances):
            pheromone_contribution = self.Q / distance
            # pheromone contribution is proportional to Q divided by the distance
            for route in routes:
                for i in range(len(route) - 1):
                    from_node = route[i]
                    to_node = route[i + 1]
                    self.pheromone_matrix[from_node][to_node] += pheromone_contribution
                    self.pheromone_matrix[to_node][from_node] += pheromone_contribution  # For undirected graph

    def calculate_total_distance(self, routes):
        """
        calculate the total distance and penalties for a set of routes

        take following paramters
        - routes: List of routes

        returns back
        - total_distance (float): the total distance including penalties
        """
        total_distance = 0.0
        penalty = 0.0
        for route in routes:
            vehicle = Vehicle(self.problem.vehicle_capacity)
            vehicle.current_location = route[0]
            vehicle.current_time = self.problem.depot.ready_time
            vehicle.load = 0
            for i in range(len(route) - 1):
                from_node = route[i]
                to_node = route[i + 1]
                if to_node != 0:
                    customer = self.problem.customers[to_node - 1]
                else:
                    customer = self.problem.depot
                travel_time = self.problem.distance_matrix[from_node][to_node]
                arrival_time = vehicle.current_time + travel_time
                if to_node != 0:
                    waiting_time = max(0, customer.ready_time - arrival_time)
                    service_begin_time = arrival_time + waiting_time
                    if service_begin_time > customer.due_date:
                        penalty += self.penalty_time * (
                            service_begin_time - customer.due_date
                        )
                    vehicle.current_time = service_begin_time + customer.service_time
                    vehicle.load += customer.demand
                    if vehicle.load > vehicle.capacity:
                        penalty += self.penalty_capacity * (
                            vehicle.load - vehicle.capacity
                        )
                else:
                    # back to depot
                    if arrival_time > self.problem.depot.due_date:
                        penalty += self.penalty_time * (
                            arrival_time - self.problem.depot.due_date
                        )
                vehicle.current_location = to_node
                total_distance += travel_time
        total_distance += penalty
        return total_distance
