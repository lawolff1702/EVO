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

    def loss(self, X, y, w=None):
        """
        Compute the binary cross-entropy loss for the logistic regression model.
        The loss is defined as:
        L(w) = -1/n * sum(y_i * log(sigmoid(X_i @ w)) + (1 - y_i) * log(1 - sigmoid(X_i @ w)))
        where n is the number of samples, X_i is the i-th sample, and y_i is the corresponding label.
        The loss is averaged over all samples.
        If w is not provided, the model's current weights are used.
        """
        if w is None:
            w = self.w
        preds = torch.clamp(self.sigmoid(X @ w), 1e-7, 1 - 1e-7) # Avoid log(0)

        #adding a term to penalize the model for low diversity in the population
        # Add diversity penalty *only* if self.optimizer is set
        if hasattr(self.optimizer, "compute_diversity"):
            diversity_term = self.optimizer.compute_diversity()
            return (-y * torch.log(preds) - (1 - y) * torch.log(1 - preds)).mean() - (self.diversity_coeff * diversity_term)
        else:
            return (-y * torch.log(preds) - (1 - y) * torch.log(1 - preds)).mean()
    
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
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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

    def loss(self, X, y, w=None):
        if w is None:
            w = self.w
        
        logits = self.forward(X, w)
        # Cross entropy --> no longer binary
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(logits, y)

        diversity_term = self.optimizer.compute_diversity() if self.optimizer else 0

        self.curr_loss = loss.item()
        self.curr_diversity = diversity_term.item() * self.diversity_coeff

        return loss - (self.diversity_coeff * diversity_term)
    
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
        # Device: default to MPS if not provided.
        self.device = device if device is not None else torch.device("mps")

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
            all_w = torch.stack(self.population)
            return all_w.std(dim=0).mean()
        else:
            raise ValueError(f"Unknown diversity metric: {metric}")

    def average_pairwise_distance(self):
        n = len(self.population)
        if n < 2:
            return 0
        dists = [torch.norm(self.population[i] - self.population[j])
                for i in range(n) for j in range(i + 1, n)]
        return torch.stack(dists).mean()
    
    def average_cosine_dissimilarity(self):
        n = len(self.population)
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                # Ensure non-zero norms
                norm_i = torch.norm(self.population[i])
                norm_j = torch.norm(self.population[j])
                if norm_i > 0 and norm_j > 0:
                    cos_sim = torch.dot(self.population[i], self.population[j]) / (norm_i * norm_j)
                else:
                    cos_sim = 0.0
                total += (1 - cos_sim)
                count += 1
        return total / count if count > 0 else 0

    def step(self, X, y):
        # Ensure X and y are on the target device.
        X = X.to(self.device)
        y = y.to(self.device)

        if len(self.population) == 0:
            # Initialize population with random weight vectors.
            self.population = [torch.rand(self.model.w.size(0), device=self.device).detach().requires_grad_(True) 
                            for _ in range(self.population_size)]
            

        # Build tuples containing (loss, unique_id, candidate) so that ties can be broken.
        pop_with_losses = [(self.model.loss(X, y, w).item(), i, w) 
                        for i, w in enumerate(self.population)]
        # Use heapq to extract the best half.
        best_half = [w for (_, _, w) in heapq.nsmallest(self.population_size // 2, pop_with_losses)]

        new_population = []
        # Generate new candidates using single-parent reproduction (with crossover)
        for _ in range(self.population_size): 
            parent1 = random.choice(best_half)
            parent2 = random.choice(best_half)

            # Inherit half from each parent
            mask = torch.rand_like(parent1) < 0.5
            child = torch.where(mask, parent1, parent2)

            mutation_mask = torch.rand_like(child) < self.mutation_rate
            # Specify device for mutation_values so it's created on the same device.
            mutation_values = torch.normal(mean=0.0, std=self.mutation_intensity,
                                        size=child.size(), device=self.device)
            child = torch.where(mutation_mask, child + mutation_values, child)

            new_population.append(child)


        for i in range(len(new_population)):
            #Load child weights into the model
            self.model.w = new_population[i].clone().detach().requires_grad_(True)
            
            # Perform one backpropagation update
            self.model.backprop_step(X, y, lr=0.05)
            
            # Save updated child back into the population
            new_population[i] = self.model.w.detach().requires_grad_(True)

        self.population = new_population
        
        pop_with_losses = [(self.model.loss(X, y, w).item(), i, w) 
                        for i, w in enumerate(new_population)]
        
        # Select the best individual from the new population using the same tie-breaker.
        best = min(pop_with_losses, key=lambda tup: (tup[0], tup[1]))[2]
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
    
def seed_dnn_from_logistic(dnn_model, logistic_model):
    """
    Seeds first layer of DNN with weights from short logistic regression training loop
    Uses logistic weights, modified with noise, and sets them in DNN's first layer
    to hopefully transfer high accuracy/low loss to start of DNN training loop
    """
    input_dim = dnn_model.layer_dims[0]
    h1_dim = dnn_model.layer_dims[1]

    # Essentially creating a copy of weight tensor which can't be modified by other processes
    logistic_weights = logistic_model.w.detach().clone()
    w_log = logistic_weights.unsqueeze(1).repeat(1, h1_dim)

    # Add random noise to each element of W
    w_log += 0.01 * torch.randn(input_dim, h1_dim, device=logistic_weights.device)

    # Set offset index to 0 to start filling neurons
    offset = 0

    # Fill all weight values with corresponding values from logistic training
    dnn_model.w.data[offset : offset + input_dim * h1_dim] = w_log.flatten()

    # Increment offset to now fill bias values (remember, working on flattened tensor)
    offset += input_dim * h1_dim

    # Fill bias values with 0 as baseline
    dnn_model.w.data[offset : offset + h1_dim] = 0