from agents.q_learner_evolver.population import Population

pop = Population(
        init_size=2,
        training_iterations=1,
        evaluation_iterations=1,
        crossover_rate=0.8,
        mutation_rate=0.2
)

while True:
    pop.one_generation()

