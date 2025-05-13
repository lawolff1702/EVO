#!/usr/bin/env python3

import os
import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from EVO import DeepNeuralNetwork, EvolutionOptimizer

def main(args):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))

    # Data pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_ds = datasets.MNIST(args.data_dir, train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=1000)

    # Model + Optimizer
    model = DeepNeuralNetwork([784, args.hidden_size, 10])
    model.use_diversity_loss = True
    optimizer = EvolutionOptimizer(model)
    optimizer.set_population_size(args.population_size)
    optimizer.use_backprop = False
    optimizer.set_diversity_coeff(args.diversity_coeff)
    optimizer.set_survivors_ratio(args.survivors_ratio)
    optimizer.set_fitness_ratio(args.fitness_ratio)
    optimizer.set_sneaker_prob(args.sneaker_prob)
    optimizer.set_mutation_intensity(args.mutation_intensity)
    optimizer.mutation_rate = args.mutation_rate

    # Load checkpoint if available
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...", flush = True)
        checkpoint = torch.load(args.checkpoint)
        model.w = checkpoint["model_w"]
        optimizer.population = checkpoint["population"]

    # Evolution loop
    for generation in range(args.generations):
        for X, y in train_loader:
            optimizer.step(X, y)

        # Test accuracy
        correct = total = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(model.device), y.to(model.device)
                preds = model.predict(X)
                correct += (preds == y).sum().item()
                total   += y.size(0)

        accuracy = 100 * correct / total
        print(f"Generation {generation+1}, Test Accuracy: {accuracy:.2f}%", flush = True)

        # Save checkpoint every few generations
        if (generation + 1) % args.checkpoint_interval == 0:
            checkpoint_path = f"checkpoint_gen_{generation+1}.pt"
            torch.save({
                "model_w": model.w,
                "population": optimizer.population
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}", flush = True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DeepNeuralNetwork on MNIST using evolutionary optimization")
    parser.add_argument("--data-dir", type=str, default=".", help="Directory for MNIST data")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--hidden-size", type=int, default=64, help="Number of hidden units")
    parser.add_argument("--population-size", type=int, default=100, help="Population size")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations")
    parser.add_argument("--diversity-coeff", type=float, default=0.1, help="Diversity coefficient")
    parser.add_argument("--survivors-ratio", type=float, default=0.1, help="Survivors ratio")
    parser.add_argument("--fitness-ratio", type=float, default=0.5, help="Fitness ratio")
    parser.add_argument("--sneaker-prob", type=float, default=0.1, help="Sneaker probability")
    parser.add_argument("--mutation-intensity", type=float, default=0.2, help="Mutation intensity")
    parser.add_argument("--mutation-rate", type=float, default=0.3, help="Mutation rate")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint file")
    parser.add_argument("--checkpoint-interval", type=int, default=5, help="Save checkpoint every N generations")
    args = parser.parse_args()

    main(args)