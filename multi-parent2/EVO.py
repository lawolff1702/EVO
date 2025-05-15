import torch
import torch.nn as nn
import random

class LinearModel:
    """
    A simple linear model for binary classification.
    """

    def __init__(self):
        self.w = None 


    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None:
            self.w = torch.rand((X.size(1)), device=X.device)
        return X @ self.w

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        score = self.score(X)
        return (score > 0).float()
    
class LogisticRegression(LinearModel):
    """
    Logistic regression model for binary classification.
    Inherits from LinearModel.
    """
    def __init__(self):
        super().__init__()
        self.diversity_coeff = 0.0
        self.optimizer = None

    def set_optimizer(self, optimizer):
        """
        Set the optimizer for the logistic regression model.
        :param optimizer: An instance of an optimizer class.
        """
        self.optimizer = optimizer
    
    def set_diversity_coeff(self, diversity_coeff):
        """
        Set the diversity coefficient for the logistic regression model.
        :param diversity_coeff: A float value representing the diversity coefficient.
        """
        self.diversity_coeff = diversity_coeff

    def sigmoid(self, x):
        """
        Compute the sigmoid function for each element in x.
        """
        return 1 / (1 + torch.exp(-x))

    def loss(self, X, y, w=None, *, include_diversity: bool = True):
        """
        Binary-cross-entropy with an optional diversity penalty.

        › The penalty is applied **only if**  
        · self.use_diversity_loss  is True  (model-level switch)            and  
        · include_diversity        is True  (call-site override)            and  
        · self.optimizer has compute_diversity().
        """
        if w is None:
            w = self.w

        # --- CE term ------------------------------------------------------
        preds = torch.clamp(self.sigmoid(X @ w), 1e-7, 1 - 1e-7)
        ce = (-y * torch.log(preds) - (1 - y) * torch.log(1 - preds)).mean()

        # --- Diversity term (optional) ------------------------------------
        diversity_active = (
            include_diversity
            and getattr(self, "use_diversity_loss", True)      # default to True if attr absent
            and hasattr(self.optimizer, "compute_diversity")
        )

        if diversity_active:
            div = self.optimizer.compute_diversity()
            return ce - self.diversity_coeff * div
        else:
            return ce
    
    def grad(self, X, y):
        """
        Compute the gradient of the loss function with respect to the weights w.
        The gradient is defined as:
        grad(w) = 1/n * sum((sigmoid(X_i @ w) - y_i) * X_i)
        where n is the number of samples, X_i is the i-th sample, and y_i is the corresponding label.
        The gradient is averaged over all samples.
        """
        sigmoid = self.sigmoid(self.score(X))
        grad = (sigmoid - y)[:, None] * X
        return grad.mean(0)
    

class DeepNeuralNetwork:
    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        self.diversity_coeff = 0.0
        self.optimizer = None
        self.curr_bce = None
        self.curr_diversity = None
        self.use_diversity_loss = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

        self.shapes = []
        total_params = 0
        for i in range(len(layer_dims) - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i+1]
            self.shapes.append((in_dim, out_dim))
            total_params += in_dim * out_dim + out_dim

        self.w = torch.rand(total_params, device=self.device, requires_grad=True)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        
    def set_diversity_coeff(self, diversity_coeff):
        self.diversity_coeff = diversity_coeff

    def forward(self, X, w=None):
        if w is None:
            w = self.w
        offset = 0
        out = X
        out = X.to(self.device)
        for in_dim, out_dim in self.shapes[:-1]:
            W = w[offset:offset + in_dim * out_dim].view(in_dim, out_dim)
            offset += in_dim * out_dim
            b = w[offset:offset + out_dim]
            offset += out_dim
            out = torch.relu(out @ W + b)

        in_dim, out_dim = self.shapes[-1]
        W = w[offset:offset + in_dim * out_dim].view(in_dim, out_dim)
        offset += in_dim * out_dim
        b = w[offset:offset + out_dim]
        logits = out @ W + b
        return logits

    def predict(self, X):
        with torch.no_grad():
            logits = self.forward(X)
            # Pick class with highest logit (probability!)
            return torch.argmax(logits, dim = 1)

    def loss(self, X, y, w=None, *, include_diversity: bool = True):
        """
        Multi-class cross-entropy with optional diversity penalty.
        Respects both self.use_diversity_loss and include_diversity.
        """
        if w is None:
            w = self.w
        X, y = X.to(self.device), y.to(self.device)

        # --- CE term ------------------------------------------------------
        logits = self.forward(X, w)
        ce = nn.CrossEntropyLoss()(logits, y)

        # --- Diversity term (optional) ------------------------------------
        diversity_active = (
            include_diversity
            and self.use_diversity_loss                       # already defined in class
            and self.optimizer is not None
        )

        if diversity_active:
            div = self.optimizer.compute_diversity()
            return ce - self.diversity_coeff * div
        else:
            return ce
    
    def backprop_step(self, X, y, lr):
        X = X.to(self.device)     # move inputs to GPU/MPS if your w lives there
        y = y.to(self.device)
        logits = self.forward(X, self.w)
        loss_function =  nn.CrossEntropyLoss()
        loss = loss_function(logits, y)

        # Go backward over each node/lead in tensor (with requires_grad=True)
        # Computes gradient for every leaf, and places the result in self.w.grad
        loss.backward()

        # Now that we are at the no-grad block, perform gradient update to weight vec
        with torch.no_grad():
            self.w -= lr * self.w.grad


        # If were to do multiple backprops in one generation.
        # need this to make sure you don't accumulate result of
        # loss.backward() into self.w.grad, and instead overwrite it
        self.w.grad.zero_()

        return loss.item()


    
import heapq

class EvolutionOptimizer():
    """
    Evolutionary algorithm optimizer for the logistic regression model.
    This optimizer uses a population of individuals (weight vectors) and evolves
    them over generations.
    """
    def __init__(self, model, device=None):
        self.model = model
        self.population = []
        self.mutation_rate = 0.1
        self.diversity_coeff = 0.5
        self.model.set_diversity_coeff(self.diversity_coeff)
        self.population_size = 100
        self.mutation_intensity = 0.1
        self.diversity_metric = "euclidean"
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.fitness_ratio = 0.5
        self.survivors_ratio = 0.1
        self.sneakers_ratio = 0.1
        self.sneaker_prob = 0.05
        self.mutation_type = "lap"
        # Temporary
        self.use_backprop = False
        self.model.set_optimizer(self)
        self.num_parents = 2

    def set_num_parents(self, num_parents):
        """
        Set the number of parents to be used for crossover.
        """
        self.num_parents = num_parents
        
    def set_use_backprop(self, use_backprop):
        self.use_backprop = use_backprop

    def set_mutation_type(self, mutation_type):
        self.mutation_type = mutation_type

    def set_fitness_ratio(self, fitness_ratio):
        self.fitness_ratio = fitness_ratio

    def set_survivors_ratio(self, survivors_ratio):
        self.survivors_ratio = survivors_ratio

    def set_sneakers_ratio(self, sneakers_ratio):
        self.sneakers_ratio = sneakers_ratio

    def set_sneaker_prob(self, sneaker_prob):
        self.sneaker_prob = sneaker_prob

    def set_mutation_rate(self, mutation_rate):
        self.mutation_rate = mutation_rate
    
    def set_mutation_intensity(self, mutation_intensity):
        self.mutation_intensity = mutation_intensity
    
    def set_population_size(self, population_size):
        self.population_size = population_size

    def set_diversity_coeff(self, diversity_coeff):
        self.diversity_coeff = diversity_coeff
        self.model.set_diversity_coeff(self.diversity_coeff)
    
    def set_diversity_metric(self, metric):
        """
        Set the diversity computation method to be used in loss calculation
        """
        self.diversity_metric = metric

    def compute_diversity(self, metric=None):
        if metric is None:
            metric = self.diversity_metric
        if metric == "euclidean":
            return self.average_pairwise_distance()
        elif metric == "cosine":
            return self.average_cosine_dissimilarity()
        elif metric == "std":
            return self.diversity_standard_deviation()
        elif metric == "variance":
            return self.average_variance()
        else:
            raise ValueError(f"Unknown diversity metric: {metric}")

    def average_pairwise_distance(self):
        n = len(self.population)
        if n < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Stack all vectors into matrix
        all_w = torch.stack(self.population)

        # Compute squared norm/magnitude for each vector (||u||^2)
        # Produces a column vector
        # all_w is matrix where each row is weight vector
        # all_w ** 2 squares each element in matrix (i.e. each weight)
        # .sum(dim=1) will sum every squared weight in each vector, giving us a 1d vector with all magnitudes squared
        # keepdim=True keeps result a column vector
        magnitude_squared = (all_w ** 2).sum(dim=1, keepdim=True)

        # Euclidean distance squared pairs identity
        # used to calculate the squared euclidean distance between all vector pairs 
        # Distance Squared between 2 vectors u and v: ||u - v||^2
        # Simplifies to ||u||^2 + ||v||^2 - 2<u,v>
        distance_squared = magnitude_squared + magnitude_squared.T - (2 * (all_w @ all_w.T))

        # Create a boolean mask fo upper triangle of matrix (values above diagonal are true)
        # https://pytorch.org/docs/stable/generated/torch.triu.html
        # Use this to avoid computing same pair twice
        # diagonal = 1 so diagonal is excluded in output
        triangle_mask = torch.triu(torch.ones(n, n, device=self.device), diagonal=1).bool()

        # Extract distances using triangle mask
        # Clamp to make sure no Os get in
        pair_distances = torch.sqrt(torch.clamp(distance_squared[triangle_mask], min=0.0))

        return pair_distances.mean()


    
    def average_cosine_dissimilarity(self):
        # Changed to vectorized
        n = len(self.population)

        # Can't perform comparison on less than 2 vectors
        if n < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Takes all individual weight vectors in population, and stacks them into single, 2D tensor
        # Gives us matrix of all weight vectors for matrix multiplication
        all_w = torch.stack(self.population)

        # Calculate magnitude of all vectors (needed for cosine similarity)
        # dim = 1 sums along features (along the row)
        # keepdim = True keeps output as 2d column vector (i.e. (n,1) rather than (n,))
        mags = torch.norm(all_w, dim=1, keepdim=True)

        # Add small small small value to avoid division by 0
        mags += 1e-8

        # Divide each weight vector by magnitude to get unit vectors
        # We want unit vectors because we don't want vector length to interfere with calculation
        # Cosine similarity is about angle between vectors, not length
        normalized = all_w / mags

        # Calculate cosine matrix
        # Mulitplying by transpose gives an matrix where each entry = cosine similarity between two vectors
        # Remember: The dot product between two vectors - what is happening in matrix multiplication - is the cos similarity of those vectors!
        # NOTE: This matrix is symmetric --> only need half (consider in next step)
        # We know this beacause the dot product is symmetric/commutative, and bottom half entries are the same as top half of matrix
        cos_matrix = normalized @ normalized.T

        # Create a boolean mask fo upper triangle of matrix (values above diagonal are true)
        # https://pytorch.org/docs/stable/generated/torch.triu.html
        # Use this to avoid computing same pair twice
        triangle_mask = torch.triu(torch.ones(n, n, device=self.device), diagonal=1).bool()
        cos_similarities = cos_matrix[triangle_mask]

        # Cosine dissimilarity = 1 - cos similarity
        cos_dissimilarities = 1 - cos_similarities

        # Return average dissimilarity across all pairs
        return cos_dissimilarities.mean()


    def average_variance(self):
        if len(self.population) < 2:
            return torch.tensor(0.0, device=self.device)
        
        all_w = torch.stack(self.population)
        variance_per_dim = all_w.var(dim=0)
        return variance_per_dim.mean()
    
    def diversity_standard_deviation(self):
        all_w = torch.stack(self.population)
        return all_w.std(dim=0).mean()


    def step(self, X, y):
        """
        One evolutionary generation.
        • Parents (best_half) are chosen by the diversity-regularised loss
          → model.loss(..., include_diversity=True)           exploration
        • Survivors (elitism) are chosen by plain CE only
          → model.loss(..., include_diversity=False)          exploitation
        """
        # -------------------------------------------------------------
        # 0. Move data to the right device and build an initial population
        # -------------------------------------------------------------
        X, y = X.to(self.device), y.to(self.device)

        if len(self.population) == 0:
            # Need parameter dimension; ensure model.w exists
            if self.model.w is None:
                _ = self.model.score(X)          # initialises self.model.w
            self.population = [
                torch.rand_like(self.model.w, device=self.device)
                for _ in range(self.population_size)
            ]

        # -------------------------------------------------------------
        # 1. Evaluate population – two different loss flavours
        # -------------------------------------------------------------
        with torch.no_grad():
            # (a) Diversity-aware loss → for parent selection
            pop_with_losses = [
                (self.model.loss(X, y, w, include_diversity=True).item(), i, w)
                for i, w in enumerate(self.population)
            ]

            # (b) CE-only loss → for elitist survivors
            pop_ce_only = [
                (self.model.loss(X, y, w, include_diversity=False).item(), i, w)
                for i, w in enumerate(self.population)
            ]

        # -------------------------------------------------------------
        # 2. Choose parents (best_half) using pop_with_losses
        # -------------------------------------------------------------
        k_parents = max(1, int(self.population_size * self.fitness_ratio))
        best_half = [
            w for (_, _, w) in heapq.nsmallest(k_parents, pop_with_losses)
        ]

        # Sneakers: occasionally let the worst performers reproduce
        if random.random() < self.sneaker_prob:
            k_sneakers = max(1, int(self.population_size * self.sneakers_ratio))
            sneakers = [
                w for (_, _, w) in heapq.nlargest(k_sneakers, pop_with_losses)
            ]
            best_half.extend(sneakers)

        # -------------------------------------------------------------
        # 3. Elitist survivors chosen **only** by CE
        # -------------------------------------------------------------
        k_survivors = max(1, int(self.population_size * self.survivors_ratio))
        survivors = [
            w for (_, _, w) in heapq.nsmallest(k_survivors, pop_ce_only)
        ]

        # -------------------------------------------------------------
        # 4. Create offspring
        # -------------------------------------------------------------
        new_population = []

        for _ in range(self.population_size - len(survivors)):
            if self.num_parents == 0:
                child = torch.rand_like(self.model.w, device=self.device)
            else:
                parents = random.sample(best_half, self.num_parents)
                child = torch.empty_like(parents[0])

                # Uniform crossover
                mask = torch.randint(0, self.num_parents, (child.numel(),),
                                     device=self.device)
                for p_idx in range(self.num_parents):
                    child[mask == p_idx] = parents[p_idx][mask == p_idx]

                # Mutation
                mutation_mask = torch.rand_like(child) < self.mutation_rate
                if self.mutation_type == "lap":
                    noise = torch.distributions.Laplace(
                        loc=0.0, scale=self.mutation_intensity
                    ).sample(child.size()).to(self.device)
                else:  # Gaussian
                    noise = torch.normal(
                        mean=0.0,
                        std=self.mutation_intensity,
                        size=child.size(),
                        device=self.device
                    )
                child = torch.where(mutation_mask, child + noise, child)

            new_population.append(child)

        # Add survivors untouched
        new_population.extend(survivors)

        # -------------------------------------------------------------
        # 5. Optional back-prop fine-tune each child
        # -------------------------------------------------------------
        for i, indiv in enumerate(new_population):
            self.model.w = indiv.to(self.device)

            if self.use_backprop:
                self.model.w.requires_grad_()
                self.model.backprop_step(X, y, lr=0.05)
                new_population[i] = self.model.w.detach()  # store detached
            else:
                new_population[i] = self.model.w

        self.population = new_population

        # -------------------------------------------------------------
        # 6. Pick champion for self.model.w  (CE-only is typical)
        # -------------------------------------------------------------
        with torch.no_grad():
            pop_ce_only = [
                (self.model.loss(X, y, w, include_diversity=False).item(), i, w)
                for i, w in enumerate(self.population)               # <- use current pop
            ]
            best = min(pop_ce_only, key=lambda t: (t[0], t[1]))[2]
        
        self.model.w = best  



class GradientDescentOptimizer():
    """
    Gradient descent optimizer for the logistic regression model.
    """
    def __init__(self, model):
        self.model = model
        self.prev_w = None  

    def step(self, X, y, alpha, beta):
        """
        Compute one step of the gradient descent update using the feature matrix X
        and target vector y.
        The update rule is:
        w_new = w_old - alpha * grad + beta * (w_old - w_prev)
        where alpha is the learning rate, beta is the momentum coefficient,
        grad is the gradient of the loss function, and w_prev is the previous weight vector.
        """
        grad = self.model.grad(X, y)

        if self.prev_w is None:
            self.prev_w = self.model.w.clone()

        momentum = beta * (self.model.w - self.prev_w)

        new_w = self.model.w - alpha * grad + momentum

        self.prev_w = self.model.w.clone()
        self.model.w = new_w