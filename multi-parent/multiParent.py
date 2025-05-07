from EVO_BASE import EvolutionOptimizer
import torch
import random

class mpEvolutionOptimizer(EvolutionOptimizer):
    def step(self, X, y, num_parents = 2):
        """
        Perform one step of the evolutionary algorithm.
        The algorithm works as follows:
        1. If the population is empty, initialize it with random weights.
        2. Sort the population based on their fitness (loss).
        3. Select the best half of the population.
        4. Create new individuals by combining genes from the best individuals.
        5. Mutate the new individuals.
        6. Update the population with the new individuals.
        7. Set the model's weights to the best individual in the population.
        """
        if self.model.population == []:
            self.model.population = [torch.rand(X.size(1)) for _ in range(self.population_size)]

        population = self.model.population
        population = sorted(population, key=lambda w: self.model.loss(X, y, w))
        population = population[:self.population_size // 2]  # Keep the best half

        new_population = []

         
        for _ in range(self.population_size): 
            # randomly select (without replacement) parents from surviving population
            parents = random.sample(population, num_parents)
            child = torch.empty_like(parents[0])

            # populate the child with genes randomly chosen from parents
            for i in range(len(child)):
                # randomly select a parent for each gene
                parent = random.choice(parents)
                child[i] = parent[i]

            mutation_mask = torch.rand_like(child) < self.mutation
            mutation_values = torch.normal(mean=0.0, std=0.1, size=child.size())
            child = torch.where(mutation_mask, child + mutation_values, child)

            new_population.append(child)

        # Update population and set model weights to best
        self.model.population = new_population
        best = min(new_population, key=lambda w: self.model.loss(X, y, w))
        self.model.w = best        

