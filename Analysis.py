import argparse
from acc_by_class import analyze_results_by_generation, aggregate_accuracies, Plotting
from hyper import *
from json_handling import *
import population, memberLKD
from population import Population
from memberLKD import Member
from sentence_transformers import SentenceTransformer, util
import numpy as np
import distances
from distances import plot_distances
import pickle

def analyze(file_path, use_classes):
    # Load the population data from the specified file
    with open(file_path, 'rb') as file:
        pop = pickle.load(file)

    # Generate embeddings for the first generation's prompt history
    first_gen = [member.prompt_history[0] for member in pop.members]
    model = SentenceTransformer("all-MiniLM-L6-v2").to('cpu')
    embeddings = model.encode(first_gen)

    # Compute cosine similarity and diversity metrics
    cos_sim = util.cos_sim(embeddings, embeddings)
    dm = 1 - cos_sim
    normalized_distance = ((dm.numpy() if hasattr(dm, 'numpy') else dm) + 1) / 2
    np.fill_diagonal(normalized_distance, 0)
    adg1 = normalized_distance.mean(axis=1)  # Average diversity metric for each individual

    # Plotting class scores or overall accuracy by generation
    Plotting(pop, use_classes)
    
    # Plot distances (optional)
    plot_distances(pop)

if __name__ == "__main__":
    # Set up argument parsing for file path and use_classes
    parser = argparse.ArgumentParser(description="Run analysis on population data.")
    parser.add_argument("--file", type=str, required=True, help="Path to the input file containing population data.")
    parser.add_argument("--use_classes", type=bool, default=True, help="Specify whether to use class-based analysis.")
    args = parser.parse_args()
    analyze(args.file, args.use_classes)#, args.experiment_num)  # Assume experiment_num is handled or removed as needed.
