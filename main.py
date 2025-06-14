from population import Population
from member import Member
from hyper import *
from utils import *
import random
import torch
from transformers import BertTokenizer, BertModel
import openai
import os
import json
import uuid
import numpy as np
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
from transformers.utils import logging
from llama_cpp import Llama
from llm import get_llm_instance  
import pickle
import copy
from collections import defaultdict
from Analysis import analyze


logging.set_verbosity_info()
logger = logging.get_logger("transformers")
logger.info("INFO")
logger.warning("WARN")


load_dotenv()
openai.api_key = ""
openai.organization = ""


# # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
# llm = Llama(
#   model_path="./mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",  # Download the model file first
#   n_ctx=2048,  # The max sequence length to use - note that longer sequence lengths require much more resources
#   n_threads=32,            # The number of CPU threads to use, tailor to your system and the resulting performance
#   n_gpu_layers=28,         # The number of layers to offload to GPU, if you have GPU acceleration available
#   chat_format = "llama-2",
#   # repetition_penalty = 2.5
# )

def main():
    logging.set_verbosity_info()
    logger = logging.get_logger("transformers")
    logger.info("INFO")
    logger.warning("WARN")
    load_dotenv()
    llm_instance = get_llm_instance()


    
    # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
    # llm = Llama(
    #   model_path="./mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",  # Download the model file first
    #   n_ctx=2048,  # The max sequence length to use - note that longer sequence lengths require much more resources
    #   n_threads=32,            # The number of CPU threads to use, tailor to your system and the resulting performance
    #   n_gpu_layers=28,         # The number of layers to offload to GPU, if you have GPU acceleration available
    #   chat_format = "llama-2",
    #   # repetition_penalty = 2.5
    # )
    
    dataset_file = 'fixed_dataset.pkl'
    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)
    
    # Retrieve variables from the fixed dataset
    task = dataset['task']
    prefix = dataset['prefix']
    examples = dataset['examples']
    answers = dataset['answers']
    pop_size = 20
    # eval_size = 50

    # Count occurrences of each target
    # target_counts = defaultdict(int)
    # for example in data['examples']:
    #     target_counts[example['target']] += 1

    # random.shuffle(data['examples'])
    # # selected_counts = defaultdict(int)
    # # max_per_target = eval_size // len(target_counts)

    # for example in data['examples']:
    #     target = example['target']
    #     # if selected_counts[target] < max_per_target:
    #     examples.append(example['input'])
    #     # options.append(list(example['target_scores'].keys()))
    #     answers.append(target)
    #     #selected_counts[target] += 1
    #     if len(examples) >= eval_size:
    #         break
    # # Shuffle data
    # random.shuffle(data['examples'])

    # # Keep track of selected samples per target
    # selected_counts = defaultdict(int)
    # max_per_target = eval_size // len(target_counts)

    # # Select samples for evaluation
    # for example in data['examples']:
    #     target = example['target']
    #     if selected_counts[target] < max_per_target:
    #         examples.append(example['input'])
    #         options.append(list(example['target_scores'].keys()))
    #         answers.append(target)
    #         selected_counts[target] += 1
    #     if len(examples) >= eval_size:
    #         break
    # task = "Given a Reddit post, correctly identify the type of personal experience with living kidney donation that the user is describing from one of the following options: Current Direct, Current Indirect, Past Direct, Past Indirect, Hypothetical/Future, General/Informational, and News/Noise"

    settings = {
        "Instruction and Reasoning Approach": ["N/A", "Step-by-Step Guidance", "Simulate Dialogue Between Two People", "Outline Common Pitfalls"],
        
        "Creativity": ["N/A", "High Creativity", "Low Creativity",  "Lateral Thinking"],
    
        "Examples Used": ["N/A", "Single Example", "Two Examples", "Three Examples"],

        "Persona": ["N/A", "Grade school math teacher", "High-achieving student", "Determined learner"]
    }

    # Create an instance of the Population class
    hyper_params = [(8, 10, 0.1), (8, 12, 0.1), (6, 8, 0.1), (6, 10, 0.1), (8, 10, 0.05), (8, 12, 0.05), (6, 8, 0.05), (6, 10, 0.05)] 
    # hyper_params = [(8, 12, 0.05), (6, 8, 0.05), (6, 10, 0.05)] #when we got stuck at 6
    # hyper_params = [(6, 8, 0.05), (6, 10, 0.05)] #restarting on 7 because it skipped the first five generations
    # hyper_params = [(6, 10, 0.05)]
    
    for num, i in enumerate(hyper_params):
        num += 1
        a, b, c = i[0], i[1], i[2]
        population = Population(num_prompts=pop_size, max_generations=20, settings=settings, task=task, prefix=prefix, 
                            problem_descriptions=examples, options=[], answers=answers)
        population.run(max_components = a, max_settings_size = b, mutation_rate = c, experiment_num = num+1)
        # with open('Results/Experiment 6/5p_hyper_8C_12S_generation5_V3.pkl', 'rb') as file:
        #     population = pickle.load(file)
        # # population.max_generations = 20
        # population.run(current_gen=5, max_components = a, max_settings_size = b, mutation_rate = c, experiment_num = num+1)
        #     #Run the genetic algorithm
        
    
        # Save the final population to a file
        with open(f'Results/Experiment {num+1}/Promptimizer_final_population_Experiment_{num+1}_v3.pkl', 'wb') as file:
            pickle.dump(population, file)
        print("Final population has been saved.")

        #run the analysis portion to get plots
        filename = f'Results/Experiment {num+1}/{int(c*100)}p_hyper_{a}C_{b}S_generation19_V3.pkl'
        
        # Call the analysis function directly
        analyze(filename, False, num)
        print(f"Analysis completed for {filename}.")
            
    

if __name__ == "__main__":
    main()
