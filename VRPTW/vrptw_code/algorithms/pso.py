import sys
import os

# corecting directory where pso.py is located
pso_dir = os.path.dirname(os.path.abspath(__file__))

# making a parent directory to sys.path to find 'models'
parent_dir = os.path.abspath(os.path.join(pso_dir, '..'))
sys.path.append(parent_dir)

import numpy as np
import random
from models.problem_instance import ProblemInstance
from models.vehicle import Vehicle

class Particle:
    def __init__(self, problem_instance, initial_position=None):
        """
        initialize a Particle for the PSO algorithm.

        take following paramters
        - problem_instance (ProblemInstance):VRPTW problem instance
        - initial_position (np.array): an optional initial position (permutation of customers)
        """
        self.problem = problem_instance
        self.dimension = self.problem.num_customers
        # initialize position as a permutation
        if initial_position is not None:
            self.position = initial_position
        else:
            self.position = np.random.permutation(self.dimension)
        self.best_position = np.copy(self.position)
        self.best_fitness = float('inf')
        self.fitness = float('inf')
        self.routes = None

    def decode(self):
        """
        decoding the particle's position (permutation) into feasible routes
        """
        customer_sequence = [idx + 1 for idx in self.position]
        unvisited = customer_sequence.copy()
        routes = []
        while unvisited:
            vehicle = Vehicle(self.problem.vehicle_capacity)
            vehicle.current_location = 0  #  depot
            vehicle.current_time = self.problem.depot.ready_time
            vehicle.load = 0
            route = [0]  # depot
            remove_indices = []
            for cust_no in unvisited:
                customer = self.problem.customers[cust_no - 1]
                travel_time = self.problem.distance_matrix[
                    vehicle.current_location
                ][cust_no]
                arrival_time = vehicle.current_time + travel_time
                # waiting until the customer's ready time if arriving early
                arrival_time = max(arrival_time, customer.ready_time)
                # checking if customer can be feasibly added
                if (
                    vehicle.load + customer.demand <= self.problem.vehicle_capacity
                    and arrival_time <= customer.due_date
                ):
                    departure_time = arrival_time + customer.service_time
                    vehicle.current_time = departure_time
                    vehicle.load += customer.demand
                    vehicle.current_location = cust_no
                    route.append(cust_no)
                    remove_indices.append(cust_no)
                else:
                    continue
            for idx in remove_indices:
                unvisited.remove(idx)
            route.append(0)  # back to depot            
            routes.append(route)
        self.routes = routes

    def evaluate_fitness(self):
        """
        evaluating the fitness of the particle based on its routes

        calculates the total distance and applies penalties for constraint violations
        """
        total_distance = 0.0
        penalty = 0.0  # taking into account of constraint violations
        penalty_time = 1000  # giving penalties for time window violations
        penalty_capacity = 1000  # or penalties when capacity violations occur
        time_window_violations = 0
        capacity_violations = 0

        for route in self.routes:
            load = 0
            time = 0.0
            current_location = route[0]  # depot
            for next_customer in route[1:]:
                if next_customer != 0:
                    customer = self.problem.customers[next_customer - 1]
                    demand = customer.demand
                    service_time = customer.service_time
                else:
                    customer = self.problem.depot
                    demand = 0
                    service_time = 0
                load += demand
                travel_time = self.problem.distance_matrix[
                    current_location
                ][next_customer]
                arrival_time = time + travel_time
                # waiting for customer's ready time if arriving early
                arrival_time = max(arrival_time, customer.ready_time)
                # checking for time window violation
                if arrival_time > customer.due_date:
                    time_penalty = penalty_time * (arrival_time - customer.due_date)
                    penalty += time_penalty
                    time_window_violations += 1
                # updating vehicle's time after servicing the customer
                time = arrival_time + service_time
                current_location = next_customer
                total_distance += travel_time
            # cheking incase load violation
            if load > self.problem.vehicle_capacity:
                capacity_penalty = penalty_capacity * (
                    load - self.problem.vehicle_capacity
                )
                penalty += capacity_penalty
                capacity_violations += 1
        self.fitness = total_distance + penalty

        if penalty > 0:
            print(
                f"Particle Penalties - Time Window Violations: {time_window_violations}, "
                f"Capacity Violations: {capacity_violations}, Total Penalty: {penalty}"
            )

    def update_velocity(self, global_best_position, w, c1, c2):
        """
        updating the particle's velocity and position

        take following paramters
        - global_best_position (np.array): global best position found so far.
        - w (float): inertia weight.
        - c1 (float): cognitive coefficient.
        - c2 (float): social coefficient.
        """
        new_position = self.position.copy()
        length = len(self.position)

        # this is for the cognitive component
        cognitive_component = self.get_swap_sequence(new_position, self.best_position)
        # calculates cognitive component as swap sequence from current to best position
        cognitive_probability = c1 / (c1 + c2)
        # probability of using cognitive component

        # this is for the social component
        social_component = self.get_swap_sequence(new_position, global_best_position)
        # calculate social component as swap sequence from current to global best position
        social_probability = c2 / (c1 + c2)
        # probability of using social component

        # combining components based on probabilities
        if random.random() < cognitive_probability:
            new_position = self.apply_swap_sequence(new_position, cognitive_component)
        if random.random() < social_probability:
            new_position = self.apply_swap_sequence(new_position, social_component)
            # applies cognitive component
        
        # using randoom swaps for inertia
        if random.random() < w:
            idx1, idx2 = np.random.choice(length, 2, replace=False)
            new_position[idx1], new_position[idx2] = new_position[idx2], new_position[idx1]
            # applies inertia by swapping two random positions

        self.position = new_position

    def get_swap_sequence(self, from_pos, to_pos):
        """
        generates a swap sequence to transform from_pos to to_pos

        take following paramters
        - from_pos (np.array): current position
        - to_pos (np.array): target position

        returns back
        - swaps: List of swap operations as tuples (i, j)
        """
        swaps = []
        temp_pos = from_pos.copy()
        for i in range(len(from_pos)):
            if temp_pos[i] != to_pos[i]:
                swap_idx = np.where(temp_pos == to_pos[i])[0][0]
                swaps.append((i, swap_idx))
                temp_pos[i], temp_pos[swap_idx] = temp_pos[swap_idx], temp_pos[i]
        return swaps

    def apply_swap_sequence(self, position, swaps):
        """
        apply a sequence of swaps to a position

        take following paramters
        - position (np.array): The position to modify
        - swaps: List of swap operations as tuples (i, j)

        returns back
        - new_position (np.array): The position after applying swaps
        """
        new_position = position.copy()
        for i, j in swaps:
            new_position[i], new_position[j] = new_position[j], new_position[i]
        return new_position

class ParticleSwarmOptimization:
    def __init__(self, problem_instance, num_particles, num_iterations, c1, c2):
        """
        initialize the Particle Swarm Optimization algorithm

        take following paramters
        - problem_instance (ProblemInstance): VRPTW problem instance
        - num_particles (int): Number of particles in the swarm
        - num_iterations (int): Number of iterations to run the algorithm
        - c1 (float): Cognitive coefficient
        - c2 (float): Social coefficient
        """
        self.problem = problem_instance
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.c1 = c1  # cognitive coefficient
        self.c2 = c2  # social coefficient
        self.swarm = self.initialize_particles()
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.best_particle = None

    def initialize_particles(self):
        """
        initializing the swarm of particles with an initial solution

        returns back
        - particles: List of Particle instances
        """
        # generating initial feasible solution using insertion heuristic
        initial_solution = self.insertion_heuristic()
        particles = []
        for _ in range(self.num_particles):
            # pertubing initial solution slightly while maintaining feasibility
            perturbed_solution = self.perturb_solution(initial_solution)
            particle = Particle(self.problem, initial_position=perturbed_solution)
            particles.append(particle)
        return particles

    def perturb_solution(self, solution):
        """
        perturb the initial solution by performing random swaps

        take following paramters
        - solution (np.array): the initial solution to perturb

        returns back
        - perturbed_solution (np.array): the perturbed solution
        """
        perturbed_solution = solution.copy()
        num_swaps = max(1, int(0.01 * len(perturbed_solution)))
        for _ in range(num_swaps):
            idx1, idx2 = np.random.choice(len(perturbed_solution), 2, replace=False)
            perturbed_solution[idx1], perturbed_solution[idx2] = (
                perturbed_solution[idx2],
                perturbed_solution[idx1],
            )
        return perturbed_solution

    def insertion_heuristic(self):
        """
        generate an initial feasible solution using an insertion heuristic

        returns back
        - initial_solution (np.array): the initial solution as a permutation of customers
        """
        unvisited = set(range(1, self.problem.num_customers + 1))
        routes = []
        while unvisited:
            route = [0]  # depot
            vehicle = Vehicle(self.problem.vehicle_capacity)
            vehicle.current_location = 0
            vehicle.current_time = self.problem.depot.ready_time
            vehicle.load = 0

            # Selecting a seed customer (earliest ready time)
            seed_customer = min(
                unvisited,
                key=lambda cust_no: self.problem.customers[cust_no - 1].ready_time,
            )
            if not self.can_insert_customer(route, seed_customer, 1):
                break  # cannot insert any customer, should start a new route
            self.insert_customer(route, seed_customer, 1)
            unvisited.remove(seed_customer)

            # when there are unvisited customers that can be inserted into the route
            while True:
                best_insertion = None
                best_insertion_cost = float('inf')
                for cust_no in unvisited:
                    for position in range(1, len(route)):  # possible insertion positions
                        if self.can_insert_customer(route, cust_no, position):
                            cost_increase = self.calculate_insertion_cost(
                                route, cust_no, position
                            )
                            if cost_increase < best_insertion_cost:
                                best_insertion = (cust_no, position)
                                best_insertion_cost = cost_increase
                if best_insertion is not None:
                    cust_no, position = best_insertion
                    self.insert_customer(route, cust_no, position)
                    unvisited.remove(cust_no)
                else:
                    break  # can no longer insert any more customers into this route
            route.append(0)  # depot
            routes.append(route)

        # converting routes into a linear sequence of customer indices
        initial_solution = []
        for route in routes:
            initial_solution.extend(
                [cust_no - 1 for cust_no in route if cust_no != 0]
            )

        return np.array(initial_solution)

    def can_insert_customer(self, route, cust_no, position):
        """
        check if a customer can be feasibly inserted at a given position in the route

        take following paramters
        - route (list): current route
        - cust_no (int): customer number to insert
        - position (int): position to insert the customer

        returns back
        - (bool): true if insertion is feasible, false otherwise.
        """
        route_copy = route[:]
        route_copy.insert(position, cust_no)
        return self.is_route_feasible(route_copy)

    def is_route_feasible(self, route):
        """
        checking if a route is feasible in terms of capacity and time windows.

        take following paramters
        - route (list): the route to check

        Returns:
        - (bool): true if route is feasible, false otherwise
        """
        load = 0
        time = 0.0
        current_location = route[0]
        for next_customer in route[1:]:
            if next_customer != 0:
                customer = self.problem.customers[next_customer - 1]
                demand = customer.demand
                service_time = customer.service_time
            else:
                customer = self.problem.depot
                demand = 0
                service_time = 0
            load += demand
            travel_time = self.problem.distance_matrix[current_location][next_customer]
            arrival_time = time + travel_time
            arrival_time = max(arrival_time, customer.ready_time)
            if arrival_time > customer.due_date:
                return False  # t window violated
            time = arrival_time + service_time
            current_location = next_customer
        if load > self.problem.vehicle_capacity:
            return False  # capacity violated
        return True

    def calculate_insertion_cost(self, route, cust_no, position):
        """
        Calculate the additional distance caused by inserting a customer at a position

        take following paramters
        - route (list): current route
        - cust_no (int): Customer number to insert
        - position (int): position to insert the customer

        returns back
        - cost_increase (float): the increase in distance due to insertion
        """
        prev_cust = route[position - 1]
        if position < len(route):
            next_cust = route[position]
        else:
            next_cust = 0  # next is depot if at end
        dist_before = self.problem.distance_matrix[prev_cust][next_cust]
        dist_after = (
            self.problem.distance_matrix[prev_cust][cust_no]
            + self.problem.distance_matrix[cust_no][next_cust]
        )
        return dist_after - dist_before

    def insert_customer(self, route, cust_no, position):
        """
        insert a customer into the route at a specific position

        take following paramters
        - route (list): Current route
        - cust_no (int): Customer number to insert
        - position (int): Position to insert the customer
        """
        route.insert(position, cust_no)

    def run(self):
        """
        Run the Particle Swarm Optimization algorithm.

        returns back
        - best_particle.routes (list): best routes found
        - global_best_fitness (float): best fitness value found
        - best_distance_history (list): tracking of best fitness values per iteration
        """
        W_MAX = 0.9
        W_MIN = 0.4
        best_distance_history = []
        for iteration in range(self.num_iterations):
            w = W_MAX - ((W_MAX - W_MIN) * iteration / self.num_iterations)
            # linearly decreasing inertia from W_MAX to W_MIN
            for particle in self.swarm:
                particle.decode()
                particle.evaluate_fitness()
                if particle.fitness < particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = np.copy(particle.position)
            # updating global best
            self.swarm.sort(key=lambda p: p.best_fitness)
            if self.swarm[0].best_fitness < self.global_best_fitness:
                self.global_best_fitness = self.swarm[0].best_fitness
                self.global_best_position = np.copy(self.swarm[0].best_position)
                self.best_particle = self.swarm[0]
            # appending the best fitness to track
            best_distance_history.append(self.global_best_fitness)
            # updating velocities and positions
            for particle in self.swarm:
                particle.update_velocity(
                    self.global_best_position, w, self.c1, self.c2
                )
            # finally printing progress every iteration
            print(
                f"Iteration {iteration+1}/{self.num_iterations}, "
                f"Best Fitness: {self.global_best_fitness:.2f}"
            )
        return self.best_particle.routes, self.global_best_fitness, best_distance_history
