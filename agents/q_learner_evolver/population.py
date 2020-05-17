import ludopy
import numpy as np

from math import inf
from copy import copy
import os

from .individual import Individual

class Population():
    def __init__(
            self,
            init_size=10,
            training_iterations=5000,
            evaluation_iterations=200,
            crossover_rate=0.8,
            mutation_rate=0.2
    ):
        self.init_size = init_size
        self.training_iterations = training_iterations
        self.evaluation_iterations = evaluation_iterations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        self.population = []

        self.current_best_fitness = -inf
        self.current_best_id = None

        self.current_id = 0

        for i in range(self.init_size):
            self.population.append(Individual(self.current_id))
            self.current_id += 1

        self.generation = 0

        os.mkdir("individuals")

    def one_generation(self):
        print("At generation " + str(self.generation) + " with ids " + str(self.population[0].id) + " to " + str(self.population[-1].id))
        should_survive = self._evaluate_population()

        parents = copy(should_survive)

        self.population = []

        for i in range(self.init_size):
            if len(should_survive) == 0:
                break

            random_int = np.random.randint(
                    0,
                    high=100
            )

            if random_int < 100 * self.crossover_rate and len(should_survive) > 1:
                individual0 = should_survive[np.random.randint(0,high=len(should_survive))]
                should_survive.remove(individual0)
                individual1 = should_survive[np.random.randint(0,high=len(should_survive))]
                should_survive.remove(individual1)

                child0, child1 = self._crossover(
                        individual0,
                        individual1

                )

                self.population.append(child0)
                self.population.append(child1)

            else:
                individual0 = should_survive[np.random.randint(0,high=len(should_survive))]
                should_survive.remove(individual0)

                child0 = self._mutate(individual0)
                self.population.append(child0)

        if len(self.population) < self.init_size:
            for i in range(self.init_size - len(self.population)):
                individual0 = parents[np.random.randint(0,high=len(parents))]

                child0 = self._mutate(individual0)
                self.population.append(child0)

        self.generation += 1

    def _evaluate_population(self):
        fitness = {}
        fitness_sum = 0

        for i in range(len(self.population)):
            self.population[i].train(training_iterations=self.training_iterations)
            fitness[i] = self.population[i].evaluate_fitness(evaluation_iterations=self.evaluation_iterations)
            fitness_sum += fitness[i]
            self.population[i].save(directory="individuals")
            if fitness[i] > self.current_best_fitness:
                self.current_best_fitness = fitness[i]
                self.current_best_id = self.population[i].id
                print("Current best:\tId: " + str(self.current_best_id) + "\tFitness: " + str(self.current_best_fitness * 100))

        should_survive = []

        for i in fitness.keys():
            random_int = np.random.randint(
                    0,
                    high=100
            )

            prop = 1 / len(self.population)

            if fitness_sum > 0:
                prop = fitness[i] / fitness_sum

            if random_int < 100 * prop:
                should_survive.append(self.population[i])

        while len(should_survive) < self.init_size / 2:
            i = list(fitness.keys())[np.random.randint(
                0,
                high=len(fitness.keys())
            )]

            if self.population[i] in should_survive:
                continue

            random_int = np.random.randint(
                    0,
                    high=100
            )

            prop = 1 / len(self.population)

            if fitness_sum > 0:
                prop = fitness[i] / fitness_sum

            if random_int < 100 * prop:
                should_survive.append(self.population[i])

        return should_survive

    def _crossover(
            self,
            individual0,
            individual1
    ):
        bits = [0,1]
        mask = []
        for i in range(13):
            mask.append(bits[np.random.randint(0,high=len(bits))])

        parameters0 = []
        parameters1 = []

        if mask[0] == 0:
            parameters0.append(self.individual0.discount_factor)
            parameters1.append(self.individual1.discount_factor)
        else:
            parameters0.append(self.individual1.discount_factor)
            parameters1.append(self.individual0.discount_factor)
        if mask[1] == 0:
            parameters0.append(self.individual0.learning_rate)
            parameters1.append(self.individual1.learning_rate)
        else:
            parameters0.append(self.individual1.learning_rate)
            parameters1.append(self.individual0.learning_rate)
        if mask[2] == 0:
            parameters0.append(self.individual0.epsilon)
            parameters1.append(self.individual1.epsilon)
        else:
            parameters0.append(self.individual1.epsilon)
            parameters1.append(self.individual0.epsilon)
        if mask[3] == 0:
            parameters0.append(self.individual0.win_reward)
            parameters1.append(self.individual1.win_reward)
        else:
            parameters0.append(self.individual1.win_reward)
            parameters1.append(self.individual0.win_reward)
        if mask[4] == 0:
            parameters0.append(self.individual0.lost_reward)
            parameters1.append(self.individual1.lost_reward)
        else:
            parameters0.append(self.individual1.lost_reward)
            parameters1.append(self.individual0.lost_reward)
        if mask[5] == 0:
            parameters0.append(self.individual0.piece_in_reward)
            parameters1.append(self.individual1.piece_in_reward)
        else:
            parameters0.append(self.individual1.piece_in_reward)
            parameters1.append(self.individual0.piece_in_reward)
        if mask[6] == 0:
            parameters0.append(self.individual0.land_on_globe_reward)
            parameters1.append(self.individual1.land_on_globe_reward)
        else:
            parameters0.append(self.individual1.land_on_globe_reward)
            parameters1.append(self.individual0.land_on_globe_reward)
        if mask[7] == 0:
            parameters0.append(self.individual0.land_on_star_reward)
            parameters1.append(self.individual1.land_on_star_reward)
        else:
            parameters0.append(self.individual1.land_on_star_reward)
            parameters1.append(self.individual0.land_on_star_reward)
        if mask[8] == 0:
            parameters0.append(self.individual0.knock_enemy_home_reward)
            parameters1.append(self.individual1.knock_enemy_home_reward)
        else:
            parameters0.append(self.individual1.knock_enemy_home_reward)
            parameters1.append(self.individual0.knock_enemy_home_reward)
        if mask[9] == 0:
            parameters0.append(self.individual0.got_knocked_home_reward)
            parameters1.append(self.individual1.got_knocked_home_reward)
        else:
            parameters0.append(self.individual1.got_knocked_home_reward)
            parameters1.append(self.individual0.got_knocked_home_reward)
        if mask[10] == 0:
            parameters0.append(self.individual0.no_move_reward)
            parameters1.append(self.individual1.no_move_reward)
        else:
            parameters0.append(self.individual1.no_move_reward)
            parameters1.append(self.individual0.no_move_reward)
        if mask[11] == 0:
            parameters0.append(self.individual0.piece_number_scale_reward)
            parameters1.append(self.individual1.piece_number_scale_reward)
        else:
            parameters0.append(self.individual1.piece_number_scale_reward)
            parameters1.append(self.individual0.piece_number_scale_reward)
        if mask[12] == 0:
            parameters0.append(self.individual0.piece_number_init_func_value)
            parameters1.append(self.individual1.piece_number_init_func_value)
        else:
            parameters0.append(self.individual1.piece_number_init_func_value)
            parameters1.append(self.individual0.piece_number_init_func_value)

        child0 = Individual(
            self.current_id,
            discount_factor=parameters0[0],
            learning_rate=parameters0[1],
            epsilon=parameters0[2],
            win_reward=parameters0[3],
            lost_reward=parameters0[4],
            piece_in_reward=parameters0[5],
            land_on_globe_reward=parameters0[6],
            land_on_star_reward=parameters0[7],
            knock_enemy_home_reward=parameters0[8],
            got_knocked_home_reward=parameters0[9],
            no_move_reward=parameters0[10],
            piece_number_scale_reward=parameters0[11],
            piece_number_init_func_value=parameters0[12],
            mutation_rate=self.mutation_rate
        )
        self.current_id += 1

        child1 = Individual(
            self.current_id,
            discount_factor=parameters1[0],
            learning_rate=parameters1[1],
            epsilon=parameters1[2],
            win_reward=parameters1[3],
            lost_reward=parameters1[4],
            piece_in_reward=parameters1[5],
            land_on_globe_reward=parameters1[6],
            land_on_star_reward=parameters1[7],
            knock_enemy_home_reward=parameters1[8],
            got_knocked_home_reward=parameters1[9],
            no_move_reward=parameters1[10],
            piece_number_scale_reward=parameters1[11],
            piece_number_init_func_value=parameters1[12],
            mutation_rate=self.mutation_rate
        )
        self.current_id += 1

        return child0, child1

    def _mutate(
            self,
            individual0
    ):
        child0 = Individual(
            self.current_id,
            discount_factor=individual0.discount_factor,
            learning_rate=individual0.learning_rate,
            epsilon=individual0.epsilon,
            win_reward=individual0.win_reward,
            lost_reward=individual0.lost_reward,
            piece_in_reward=individual0.piece_in_reward,
            land_on_globe_reward=individual0.land_on_globe_reward,
            land_on_star_reward=individual0.land_on_star_reward,
            knock_enemy_home_reward=individual0.knock_enemy_home_reward,
            got_knocked_home_reward=individual0.got_knocked_home_reward,
            no_move_reward=individual0.no_move_reward,
            piece_number_scale_reward=individual0.piece_number_scale_reward,
            piece_number_init_func_value=individual0.piece_number_init_func_value,
            mutation_rate=self.mutation_rate
        )
        self.current_id += 1

        return child0
