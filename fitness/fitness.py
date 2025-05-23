import sys
sys.path.append('../ExplorationVsExploitation')
from EVO import EvolutionOptimizer
import torch
import random
import heapq
from torch import nn

class FitnessOptimizer(EvolutionOptimizer):
    """
    Fitness Optimizer class that inherits from EvolutionOptimizer.
    """
    def __init__(self, model):
        super().__init__(model)
        self.model = model
        self.population = []
        self.mutation_rate = 0.1
        self.diversity_coeff = 0.5
        self.model.set_diversity_coeff(self.diversity_coeff)
        self.population_size = 100
        self.mutation_intensity = 0.1
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.fitness_ratio = 0.5
        self.survivors_ratio = 0.1
        self.sneakers_ratio = 0.1
        self.sneaker_prob = 0.05
        self.mutation_type = "lap"

    def set_fitness_ratio(self, fitness_ratio):
        self.fitness_ratio = fitness_ratio

    def set_survivors_ratio(self, survivors_ratio):
        self.survivors_ratio = survivors_ratio

    def set_sneakers_ratio(self, sneakers_ratio):
        self.sneakers_ratio = sneakers_ratio

    def set_sneaker_prob(self, sneaker_prob):
        self.sneaker_prob = sneaker_prob
    
    def ce_loss(self, X, y, w=None):
        if w is None:
            w = self.w
        
        logits = self.model.forward(X, w)

        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(logits, y)

        self.curr_loss = loss.item()

        return loss

    def step(self, X, y):
        # Ensure X and y are on the target device.
        X = X.to(self.device)
        y = y.to(self.device)

        if len(self.population) == 0:
            # Initialize population with random weight vectors.
            self.population = [torch.rand(X.size(1), device=self.device) 
                            for _ in range(self.population_size)]

        # Build tuples containing (loss, unique_id, candidate) so that ties can be broken.
        pop_with_losses = [(self.model.loss(X, y, w).item(), i, w) 
                        for i, w in enumerate(self.population)]
        
        pop_with_CE = [(self.ce_loss(X, y, w), i, w)
                        for i, w in enumerate(self.population)]
        
        # Use heapq to extract the best population based on the fitness threshold.
        best_half = [w for (_, _, w) in heapq.nsmallest(int(self.population_size * self.fitness_ratio), pop_with_losses)]

        # Low-ranking candidates are added to the best_half with low probability.
        if random.random() < self.sneaker_prob:
            sneakers = [w for (_, _, w) in heapq.nlargest(int(self.population_size * self.sneakers_ratio), pop_with_losses)]
            best_half.extend(sneakers)
        
        survivors = [w for (_, _, w) in heapq.nsmallest(int(self.population_size * self.survivors_ratio), pop_with_CE)]


        new_population = []
        # Generate new candidates using single-parent reproduction (with crossover)
        for _ in range(self.population_size - len(survivors)): 
            parent1 = random.choice(best_half)
            parent2 = random.choice(best_half)

            # Inherit half from each parent
            mask = torch.rand_like(parent1) < 0.5
            child = torch.where(mask, parent1, parent2)

            mutation_mask = torch.rand_like(child) < self.mutation_rate
            # Specify device for mutation_values so it's created on the same device.
            if self.mutation_type == "lap":
                mutation_values = torch.distributions.Laplace(
                    loc=0.0, scale=self.mutation_intensity
                ).sample(child.size()).to(self.device)
            else:
                mutation_values = torch.normal(
                    mean=0.0,
                    std=self.mutation_intensity,
                    size=child.size(),
                    device=self.device
                )

            child = torch.where(mutation_mask, child + mutation_values, child)

            new_population.append(child)

        # Add survivors to the new population.
        new_population.extend(survivors)

        self.population = new_population
        # Select the best individual from the new population using the same tie-breaker.
        pop_with_losses = [(self.model.loss(X, y, w).item(), i, w) 
                        for i, w in enumerate(new_population)]
        best = min(pop_with_losses, key=lambda tup: (tup[0], tup[1]))[2]
        self.model.w = best