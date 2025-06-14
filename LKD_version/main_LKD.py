from populationLKD import Population
from memberLKD import Member
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

    
    

    dataset = pd.read_json('LKD_experiences_modified.json')
    # dataset_file = 'LKD_3_Classes_dataset.pkl'
    # with open(dataset_file, 'rb') as f:
    #     dataset = pickle.load(f)
    # Retrieve variables from the fixed dataset
    task = dataset['task_prefix']
    prefix = dataset['task_prefix']
    task_type = dataset['task_type']
    pop_size = 20
    # eval_size = 90

    example_indices = [0,1,2,3,4,6,7,8,9,10,11, 12, 13, 16, 21, 25, 32, 37, 38, 40, 41, 42, 46, 50, 52, 56, 57, 63, 67, 68, 69, 71, 74,
                       523, 524, 525, 526, 527, 531, 532, 533, 534, 537, 539, 543, 550, 559, 560, 563, 574, 571, 569, 595, 602, 614, 
                       624, 903, 904, 979, 1116, 1117, 1119, 1120, 1121, 1122, 1126, 1569, 1568, 1567, 1566, 1564, 1563, 1562, 1561, 
                       1558, 1556, 1553, 1549, 1548, 1547, 1546, 1544, 1541, 1540, 1538, 1537, 1532, 1526, 1531, 1525, 1524, 1522, 1515, 
                       1514, 1513, 1511, 1510, 1494, 1492]
    #Count occurrences of each target
    # target_counts = defaultdict(int)
    # for example in dataset['examples']:
    #     target_counts[example['target']] += 1

    # random.shuffle(dataset['examples'])
    # selected_counts = defaultdict(int)
    # max_per_target = eval_size // len(target_counts)

    # for example in dataset['examples']:
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

    # Keep track of selected samples per target
    # selected_counts = defaultdict(int)
    # max_per_target = eval_size // len(target_counts)

    # Select samples for evaluation
    examples = []
    options = []
    answers = []
    for idx in example_indices:#dataset['examples']:
        example = dataset['examples'][idx]
        target = example['target']
        # if selected_counts[target] < max_per_target:
        examples.append(example['input'])
        options.append(list(example['target_scores'].keys()))
        answers.append(target)
        #     selected_counts[target] += 1
        # if len(examples) >= eval_size:
        #     break

 
    settings = {
        "Past Classification Definition": [
            "N/A",
            "The person is telling a previous story about themselves or someone with whom they have a personal connection.",
            "Content contains explicit past tense wording",
            "Include posts where users reflect deeply on the donation experience",
            "The person is inviting questions about their donation experience"
        ],
        "Present Classification Definition": [
            "N/A",
            "Focus on 'right now' indicators",
            "Emphasize action/ongoing process",
            "Require explicit present-tense words",
            "The person, or their loved one, is currently dealing with a transplant decision"
        ],
        "Other Classification Definition": [
            "N/A",
            "News & hypothetical discussions",
            "Identify speculative or non-personal content",
            "Catch-all for unclear posts",
            "Capture posts discussing donation but with no personal connection"
        ],
        "Personas of two people who are dialoguing until they reach a classification consensus": [
            "N/A and N/A",
            "Transplant social worker and data scientist",
            "AI classifier and nephrologist",
            "Experienced kidney donor and superintelligent AI",
            "Patient engagement expert and new employee"
        ],
        "Other helpful content and context":[
            "N/A",
            "Transtheoretical model or states of change model",
            "Lived experience narratives from transplant forums",
            "Temporal cue words (e.g., 'last year', 'currently', 'maybe someday')",
            "Decision-making frameworks in health communication"
        ]
    }


    
    
    #up to 6 components, 8 settings, and a hyper mutation rate of 5%
    a, b, c = 6, 8, 0.05
    population = Population(num_prompts=pop_size, max_generations=20, settings=settings, task=task, prefix=prefix, task_type = task_type,
                        problem_descriptions=examples, options=options, answers=answers)
    population.run(max_components = a, max_settings_size = b, mutation_rate = c)
    # with open('Results/LKD/3_classes_v3_generation19.pkl', 'rb') as file:
        # population = pickle.load(file)
    # population.max_generations = 20
    # population.run(current_gen=3, max_components = a, max_settings_size = b, mutation_rate = c)
    #     #Run the genetic algorithm
    # Save the final population to a file
    with open(f'Results/LKD/3_classes_v3.pkl', 'wb') as file:
        pickle.dump(population, file)
    print("Final population has been saved.")

    #run the analysis portion to get plots
    filename = f'Results/LKD/3_classes_v3.pkl'
    
    # Call the analysis function directly
    analyze(filename, True)
    print(f"Analysis completed for {filename}.")

    

if __name__ == "__main__":
    main()
