from collections import defaultdict
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    UserMessage,
    ToolMessage
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import random
import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer
from scipy.spatial.distance import cosine
import openai
import os
import json
import uuid
import sys
import numpy as np
from dotenv import load_dotenv
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from transformers.utils import logging
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama
import re
import pickle
import copy
print('finished importing!')

t_s = ["How could I devise an experiment to help solve that problem?",
    "Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
    "How could I measure progress on this problem?",
    "How can I simplify the problem so that it is easier to solve?",
    "What are the key assumptions underlying this problem?",
    "What are the potential risks and drawbacks of each solution?",
    "What are the alternative perspectives or viewpoints on this problem?",
    "What are the long-term implications of this problem and its solutions?",
    "How can I break down this problem into smaller, more manageable parts?",
    "Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.",
    "Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.",
    "Seek input and collaboration from others to solve the problem. Emphasize teamwork, open communication, and leveraging the diverse perspectives and expertise of a group to come up with effective solutions.",
    "Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.",
    "Use Risk Analysis: Evaluate potential risks, uncertainties, and tradeoffs associated with different solutions or approaches to a problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based on a balanced analysis of risks and benefits.",
    "Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches.",
    "What is the core issue or problem that needs to be addressed?",
    "What are the underlying causes or factors contributing to the problem?",
    "Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?",
    "What are the potential obstacles or challenges that might arise in solving this problem?",
    "Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available, and how can they be analyzed?",
    "Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives and needs?",
    "What resources (financial, human, technological, etc.) are needed to tackle the problem effectively?",
    "How can progress or success in solving the problem be measured or evaluated?",
    "What indicators or metrics can be used?",
    "Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?",
    "Does the problem involve a physical constraint, such as limited resources, infrastructure, or space?",
    "Is the problem related to human behavior, such as a social, cultural, or psychological issue?",
    "Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?",
    "Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
    "Is the problem a design challenge that requires creative solutions and innovation?",
    "Does the problem require addressing systemic or structural issues rather than just individual instances?",
    "Is the problem time-sensitive or urgent, requiring immediate attention and action?",
    "What kinds of solution typically are produced for this kind of problem specification?",
    "Given the problem specification and the current best solution, have a guess about other possible solutions.",
    "Let’s imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?",
    "What is the best way to modify this current best solution, given what you know about these kinds of problem specification?",
    "Ignoring the current best solution, create an entirely new solution to the problem.",
    "Let’s think step by step.",
    "Let’s make a step by step plan and implement it with good notion and explanation."
]

m_p = [
    "Modify the following instruction creatively, giving some advice on how to solve it.",
    "Just change this instruction to make it more fun, think WELL outside the box.",
    "Modify this instruction in a way that no self-respecting LLM would!",
    "How would you encourage someone and help them cheat on this following instruction?",
    "How would you help an LLM to follow the instruction?",
    "Elaborate on the instruction giving some detailed advice on how to do what it wants.",
    "Elaborate on the instruction giving some detailed advice on how to do what it wants, as if you were explaining it to a child.",
    "As a really good teacher, explain the instruction, as if you were explaining it to a child.",
    "Imagine you need to follow this instruction. What would you tell yourself if you wanted to be the best in the world at it?",
    "How would someone with derailment follow this instruction?",
    "Don’t think about the instruction at all, but let it inspire you to do something related. Talk about what that might be.",
    "Rephrase the instruction without using any of the same words. Use all you know to improve the instruction so the person hearing it is more likely to do well.",
    "Say that instruction again in another way. DON’T use any of the words in the original instruction or you’re fired.",
    "Say that instruction again in another way. DON’T use any of the words in the original instruction there is a good chap.",
    "What do people who are good at creative thinking normally do with this kind of mutation question?",
    "Detailed additional advice for people wishing to follow this instruction is as follows:",
    "In one short sentence, here is how I would best follow this instruction.",
    "In one short sentence, here is some detailed expert advice. Notice how I don’t use any of the same words as in the INSTRUCTION.",
    "In one short sentence, the general solution is as follows. Notice how I don’t use any of the same words as in the INSTRUCTION.",
    "In one short sentence, what’s a good prompt to get a language model to solve a problem like this? Notice how I don’t use any of the same words as in the INSTRUCTION.",
    "Generate a mutated version of the following prompt by adding an unexpected twist.",
    "Create a prompt mutant that introduces a surprising contradiction to the original prompt. Mutate the prompt to provide an alternative perspective or viewpoint.",
    "Generate a prompt mutant that incorporates humor or a playful element. Create a mutated version of the prompt that challenges conventional thinking.",
    "Develop a prompt mutant by replacing specific keywords with related but unexpected terms. Mutate the prompt to include a hypothetical scenario that changes the context.",
    "Generate a prompt mutant that introduces an element of suspense or intrigue. Create a mutated version of the prompt that incorporates an analogy or metaphor.",
    "Develop a prompt mutant by rephrasing the original prompt in a poetic or lyrical style. Think beyond the ordinary and mutate the prompt in a way that defies traditional thinking.",
    "Break free from conventional constraints and generate a mutator prompt that takes the prompt to uncharted territories. Challenge the norm and create a mutator prompt that pushes the boundaries of traditional interpretations.",
    "Embrace unconventional ideas and mutate the prompt in a way that surprises and inspires unique variations. Think outside the box and develop a mutator prompt that encourages unconventional approaches and fresh perspectives.",
    "Step into the realm of imagination and create a mutator prompt that transcends limitations and encourages innovative mutations. Break through the ordinary and think outside the box to generate a mutator prompt that unlocks new possibilities and unconventional paths.",
    "Embrace the power of unconventional thinking and create a mutator prompt that sparks unconventional mutations and imaginative outcomes. Challenge traditional assumptions and break the mold with a mutator prompt that encourages revolutionary and out-of-the-box variations.",
    "Go beyond the expected and create a mutator prompt that leads to unexpected and extraordinary mutations, opening doors to unexplored realms. Increase Specificity: If the original prompt is too general, like ’Tell me about X,’ the modified version could be, ’Discuss the history, impact, and current status of X.’",
    "Ask for Opinions/Analysis: If the original prompt only asks for a fact, such as ’What is X?’, the improved prompt could be, ’What is X, and what are its implications for Y?’",
    "Encourage Creativity: For creative writing prompts like ’Write a story about X,’ an improved version could be, ’Write a fantasy story about X set in a world where Y is possible.’",
    "Include Multiple Perspectives: For a prompt like ’What is the impact of X on Y?’, an improved version could be, ’What is the impact of X on Y from the perspective of A, B, and C?’",
    "Request More Detailed Responses: If the original prompt is ’Describe X,’ the improved version could be, ’Describe X, focusing on its physical features, historical significance, and cultural relevance.’",
    "Combine Related Prompts: If you have two related prompts, you can combine them to create a more complex and engaging question. For instance, ’What is X?’ and ’Why is Y important?’ could be combined to form ’What is X and why is it important in the context of Y?’",
    "Break Down Complex Questions: If a prompt seems too complex, like ’Discuss X,’ the improved version could be, ’What is X? What are its main characteristics? What effects does it have on Y and Z?’",
    "Use Open-Ended Questions: Instead of ’Is X true?’, you could ask, ’What are the arguments for and against the truth of X?’",
    "Request Comparisons: Instead of ’Describe X,’ ask ’Compare and contrast X and Y.’",
    "Include Context: If a prompt seems to lack context, like ’Describe X,’ the improved version could be, ’Describe X in the context of its impact on Y during the Z period.’",
    "Make the prompt more visual: Ask the user to visualize the problem or scenario being presented in the prompt.",
    "Ask for a thorough review: Instead of just presenting the problem, ask the user to write down all the relevant information and identify what’s missing.",
    "Invoke previous experiences: Modify the prompt to ask the user to recall a similar problem they’ve successfully solved before.",
    "Encourage a fresh perspective: Suggest in your prompt that the user take a moment to clear their mind before re-approaching the problem.",
    "Promote breaking down problems: Instead of asking the user to solve the problem as a whole, prompt them to break it down into smaller, more manageable parts.",
    "Ask for comprehension: Modify the prompt to ask the user to review and confirm their understanding of all aspects of the problem.",
    "Suggest explanation to others: Change the prompt to suggest that the user try to explain the problem to someone else as a way to simplify it.",
    "Prompt for solution visualization: Instead of just asking for the solution, encourage the user to imagine the solution and the steps required to get there in your prompt.",
    "Encourage reverse thinking: Improve the prompt by asking the user to think about the problem in reverse, starting with the solution and working backwards.",
    "Recommend taking a break: Modify the prompt to suggest that the user take a short break, allowing their subconscious to work on the problem.",
    "What errors are there in the solution?",
    "How could you improve the working out of the problem?",
    "Look carefully to see what you did wrong, how could you fix the problem?",
    "CORRECTION =",
    "Does the above text make sense? What seems wrong with it? Here is an attempt to fix it:",
    "The above working out has some errors, here is a version with the errors fixed."
]


# filename = 'my_object.pkl'

logging.set_verbosity_info()
logger = logging.get_logger("transformers")
logger.info("INFO")
logger.warning("WARN")


load_dotenv()
openai.api_key = ""
openai.organization = ""


# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = Llama(
  model_path="./mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",  # Download the model file first
  n_ctx=2048,  # The max sequence length to use - note that longer sequence lengths require much more resources
  n_threads=48,            # The number of CPU threads to use, tailor to your system and the resulting performance
  n_gpu_layers=0,         # The number of layers to offload to GPU, if you have GPU acceleration available
  chat_format = "llama-2",
  # repetition_penalty = 2.5
)
llm.verbose = False


def Initialize(t_s, m_p, task, ducc = False):
    initialized_prompt = None
#     for i in range(num_prompts):
    if ducc:
        ts = "Expert 1 and Expert 2, you two are experts in living kidney donation and patient decision making. Dialogue until you reach a consensus on the given task." 
    else:
        ts = np.random.choice(t_s)
    mp = np.random.choice(m_p)

    response = llm(f":INSTRUCTION: We are creating a mutated prompt that solves a problem given the following information.\
                    Thinking style/Given prompt: {ts} \
                    Mutation instructions: {mp} \
                    Problem to solve: {task} \
                    OUTPUT:",
                max_tokens = 1024, 
                # frequency_penalty=0.7,
                stop = ['INSTRUCTION'],
                echo = False)
                    
            
    x = response['choices'][0]['text']
    print("INITIALIZED RESULT", x)
    return x
    

class Population:
    def __init__(self, num_prompts, max_generations, t_s, m_p, task, prefix, problem_descriptions, answers, options,**kwargs):
        self.thinking_styles = t_s
        self.mutation_prompts = m_p
        self.members = [Member(t_s, m_p, task, ducc=True) if i<5 else Member(t_s, m_p, task, ducc=False) for i in range(num_prompts) ]
        self.generation = 0
        self.max_generations = max_generations
        self.best_members = []
        self.global_best_prompt = []
        # self.best_DFR_history = []
        self.elite_history = []
        self.global_best_score = [0]
        self.similarity_limit = 0.90
        self.task = task
        self.prefix = prefix
        self.problem_descriptions = problem_descriptions
        self.options = options
        self.answers = answers
        self.promptimal_frontier = [member for member in self.members] #self.members.copy
        self.promptimal_frontier_history = []
        self.accuracy_threshold = 0.1
        self.distance_matrix = np.ones((len(self.members), len(self.members)))
        self.norm_dm = self.distance_matrix
        
        
    #UPDATE THIS BASED ON DFR
    def evaluate_members(self):
        best_member = None
        best_member_score = 0
        self.get_DFR()
        for member in self.members:
            if member.was_mutated:
                member.score = member.evaluate_prompt(self)  # Placeholder for prompt evaluation function
            member.accuracy_history.append(member.score)
            # print(member.score, min(self.global_best_score))
            if member.score > min(self.global_best_score):
                self.global_best_prompt = [member.prompt]
                self.global_best_score = [member.score]
            elif member.score== self.global_best_score:
                self.global_best_prompt.append(member.prompt)
                self.global_best_score.append(member.score)
            if member.score > best_member_score:
                best_member_score = member.score
                best_member = member
            # if member.DFR > best_member.DFR:
                member.best_ancestors.append((best_member.prompt, best_member.score))
        self.elite_history.append((self.generation, best_member, best_member_score))

    def normalize_distance_matrix(self):
        distance_matrix = np.array(self.distance_matrix)
        min_distance = np.min(distance_matrix)
        max_distance = np.max(distance_matrix)
        self.norm_dm = (self.distance_matrix - min_distance) / (max_distance - min_distance)
        
    def population_distances(self):
        model = SentenceTransformer("all-MiniLM-L6-v2").to('cpu')
        sentences = [a.prompt for a in self.members]
        
        # Encode all sentences
        embeddings = model.encode(sentences)
        
        # Compute cosine similarity between all pairs
        cos_sim = util.cos_sim(embeddings, embeddings)
        self.distance_matrix = 1 - cos_sim

    def adjust_weights(self):
        # Dynamic weight adjustment based on the generation number
        # Early in the generations, prioritize diversity; later, prioritize fitness
        progress_ratio = self.generation / self.max_generations
        sigmoid_progress = 1 / (1 + np.exp(-10 * (progress_ratio - 0.5)))
        exponential_progress = progress_ratio**2

        #choose the ratio method to use. We can experiment with these
        r = progress_ratio #sigmoid_progress # exponential_progress
        weight_fitness = r  # Increases over time
        weight_diversity = 1 - r  # Decreases over time
        return weight_fitness, weight_diversity
    
    #create the diversity-fitness ratio
    def get_DFR(self):
        fitness = np.array([member.score for member in self.members])/105
        # for i in self.members:
        # normalized_fitness = fitness / 100
        
        # Normalize distance values to [0, 1]
        # Ensure distance_matrix is a NumPy array
        normalized_distance = ((self.distance_matrix.numpy() if hasattr(self.distance_matrix, 'numpy') else self.distance_matrix) + 1) / 2
        np.fill_diagonal(normalized_distance, 0)  # Now this should work

        # Calculate average distance (diversity) for each individual
        average_distance = normalized_distance.mean(axis=1)
        
        # Get dynamic weights
        weight_fitness, weight_diversity = self.adjust_weights()
        
        # Calculate the Dynamic Diversity-Fitness Ratio (DFR) for each individual
        DFR = (weight_fitness * fitness) + (weight_diversity * average_distance)
        for i, member in enumerate(self.members):
            member.DFR = DFR[i]
            member.DFR_history.append(DFR[i])
            member.avg_distance = average_distance[i]
            member.avg_distance_history.append(average_distance[i])
        # return DFR
    
    def filter_by_score(self, min_score):
        return [member for member in self.members if member.score >= min_score]
    
    def update_promptimal_frontier(self):
        self.population_distances()
        self.normalize_distance_matrix()
        
        current_max_score = max(member.score for member in self.members)
        frontier_max_score = max((member.score for member in self.promptimal_frontier), default=0)  # default=0 handles empty frontier
        max_score = max(current_max_score, frontier_max_score)
        min_score = max_score * (1 - self.accuracy_threshold)

        self.promptimal_frontier = [member for member in self.promptimal_frontier if member.score >= min_score]
        new_candidates = self.filter_by_score(min_score)
        
        if not self.promptimal_frontier and new_candidates:
            # Start with the most distant candidate
            initial_candidate = max(new_candidates, key=lambda member: np.mean([self.distance_matrix[self.members.index(member), self.members.index(other)] for other in new_candidates if member != other]))
            self.promptimal_frontier.append(initial_candidate)
            new_candidates.remove(initial_candidate)
        
        while new_candidates:
            for candidate in new_candidates[:]:
                candidate_index = self.members.index(candidate)
                is_dissimilar = all(self.distance_matrix[candidate_index, self.members.index(frontier_member)] >= self.similarity_limit for frontier_member in self.promptimal_frontier)
                
                if is_dissimilar:
                    self.promptimal_frontier.append(candidate)
                    new_candidates.remove(candidate)
                else:
                    new_candidates.remove(candidate)
        
        self.promptimal_frontier_history.append((self.generation, self.promptimal_frontier.copy()))
    
                
    def binary_tournament(self, available_members):
        contestants = random.sample(available_members, 2)
        winner, loser = sorted(contestants, key=lambda x: x.score, reverse=True)
        return winner, loser

    def run(self, current_gen=None):
        if current_gen is not None:
            a, b = current_gen+1, self.max_generations
        else:
            a, b = 0, self.max_generations
        for _ in range(a, b):
            self.generation = _
            self.evaluate_members()
            available_members = self.members.copy()  # Create a copy of the member list for this generation
            while len(available_members) >= 2:  # Ensure there are at least two members for a tournament
                print(f'Generation {_}, there are {len(available_members)} left.')
                winner, loser = self.binary_tournament(available_members)
                # print('Winner prompt to be mutated', winner.prompt)
                loser.mutate(winner.prompt, winner.id, self)  # Winner's ID is passed as the parent
                winner.was_mutated = False
                winner.mutation_history.append(False)
                loser.was_mutated = True
                loser.mutation_history.append(True)
                # Remove the selected members from the available pool to prevent repeat tournaments
                available_members.remove(winner)
                available_members.remove(loser)
                
#                 self.update_promptimal_frontier(winner)
            self.best_members.append((self.generation, self.global_best_prompt, self.global_best_score))
            self.update_promptimal_frontier()
            
            # Use `with` statement to ensure proper resource management
            print("DUMPING PICKLE")
            print('\n'*5)
            with open(f'V2_coarse_grained_105_eval_20_gens_50_members_ducc_bias_{self.generation}.pkl', 'wb') as file:
                pickle.dump(self, file)
            print("PICKLE HAS BEEN DUMPED")
            #sys.exit()
#     def mutate(self, prompt):
#         # Placeholder for mutation logic
#         # Modify the prompt slightly and return the new prompt
#         return prompt + " mutation"

class Member:
    def __init__(self, t_s, m_p, task, ducc, **kwargs):
        print('creating a member')
        self.id = str(uuid.uuid4())
        self.prompt = Initialize(t_s, m_p, task, ducc)
        self.score = 0 #self.evaluate_prompt(population)
        self.prompt_history = [self.prompt]
        self.answer_tracker = []
        self.accuracy_history = []
        self.best_prompt = copy.deepcopy(self.prompt)
        self.location = []#vectorize 
        self.best_ancestors = []
        self.avg_distance = None
        self.avg_distance_history = []
        self.DFR = None
        self.DFR_history = []
        self.tokenizer = MistralTokenizer.v1()
        self.unacceptable_counts = 0
        self.was_mutated = False
        self.mutation_history = [False]
        
    def __getstate__(self):
        # Exclude the tokenizer from the state to be pickled
        state = self.__dict__.copy()
        if 'tokenizer' in state:
            del state['tokenizer']
        return state

    def __setstate__(self, state):
        # Restore the state and reinitialize the tokenizer
        self.__dict__.update(state)
        self.tokenizer = MistralTokenizer.v1()
        
    def truncate_to_fit(self, text, max_tokens):
        # tokenizer = MistralTokenizer.v1()
        completion_request = ChatCompletionRequest(messages=[UserMessage(content=text)])
        tokens = self.tokenizer.encode_chat_completion(completion_request)
        if len(tokens.tokens) > max_tokens:
            tokens = tokens.tokens[:max_tokens]
            return self.tokenizer.decode(tokens) + '...'
        return text

    def truncate_list(self, prompts_sorted, max_tokens):
        # tokenizer = MistralTokenizer.v1()
        total_tokens = 0
        truncated_list = []
        for prompt in prompts_sorted:
            if not isinstance(prompt, str) or not prompt.strip():
                print(f"invalid prompt encountered: {prompt}")
                continue
            
            completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
            prompt_tokens = len(self.tokenizer.encode_chat_completion(completion_request).tokens)
            if total_tokens + prompt_tokens <= max_tokens:
                truncated_list.append(prompt)
                total_tokens += prompt_tokens
            else:
                break
        return truncated_list

    def prompt_proofing(self, prompt, population, counter = None):
        task = population.task
        p = self.prompt
        response = llm(f": You will be given a prompt, or written instruction, that aims to help an AI language model to accomplish this task: {task}.\
                            Due to the stochastic nature of our larger process, the prompts we are given are sometimes very poor for this task, \
                            and you must help us determine whether the given prompt will be of any use. It is fine if a prompt is of relatively low quality,\
                            but we must remove prompts which are completely useless. You will judge whether the prompt is acceptable.\
                            This is the prompt:{p}. Please clearly indicate ACCEPTABLE or UNACCEPTABLE.",
                   max_tokens = 1024, 
                   stop = ['INSTRUCTION:'],
                   # frequency_penalty=0.7,
                   echo = False)

        x = response['choices'][0]['text']
        # print("EVALUATION RESULT:", x)
        #self.prompt_history.append(x)
        # Extract the answer
        label_pattern = re.compile(r"(ACCEPTABLE|UNACCEPTABLE)", re.IGNORECASE)
        
        # Find all matching labels in the response
        matches = label_pattern.findall(x)
        if matches:
            res = matches[-1]
        else:
            res = None
        if res and res == "ACCEPTABLE":
            return True
        elif res and res == "UNACCEPTABLE":
            self.unacceptable_counts += 1
            return False
        elif not res:
            if not counter:
                counter = 1
                self.prompt_proofing(p, population, counter)
            elif counter <=3:
                counter += 1
                self.prompt_proofing(p, population, counter)
            else:
                return True
             
    def normalize_answer(self, value):
        # Remove currency symbols and other non-numeric characters
        value = re.sub(r'[\$,]', '', value)
        
        # Convert words like "million", "billion" to numerical equivalents
        value = value.lower()
        if "million" in value:
            value = value.replace("million", "")
            value = float(value) * 1e6
        elif "billion" in value:
            value = value.replace("billion", "")
            value = float(value) * 1e9
        elif "thousand" in value:
            value = value.replace("thousand", "")
            value = float(value) * 1e3
        else:
            try:
                # Convert to float or int
                value = float(value)
                if value.is_integer():
                    value = int(value)
            except ValueError:
                pass  # If conversion fails, leave the value as is
        
        # Convert to a string without unnecessary decimals
        if isinstance(value, float) and value.is_integer():
            value = int(value)
    
        return str(value).strip()
    def replace_large_numbers(self, text):
        # Define a pattern that captures numbers followed by 'million', 'billion', 'thousand', or currency symbols
        pattern = r'(\$?\d+(?:\.\d+)?(?:\s?(million|billion|thousand))?)'
        
        def convert_match(match):
            value = match.group(0).lower()
            number = re.sub(r'[^\d.]', '', value)  # Remove non-numeric characters (except decimal point)
            
            if 'million' in value:
                number = float(number) * 1e6
            elif 'billion' in value:
                number = float(number) * 1e9
            elif 'thousand' in value:
                number = float(number) * 1e3
            else:
                number = float(number)
            
            # Convert to integer if applicable
            if number.is_integer():
                number = int(number)
            
            return str(int(number))  # Return the number as an integer string
        
        # Replace matches in the text using the convert_match function
        return re.sub(pattern, convert_match, text)    
        
    def evaluate_prompt(self, population):
        accuracy = 0
        task = population.task
        # Define maximum tokens for each part to ensure the total is within the limit
        total_max_tokens = 2000

        # Allocate tokens for each part, ensuring the total does not exceed the limit
        max_tokens_prompt = 500
        max_tokens_option = 25
        # tokenizer = MistralTokenizer.v1()
        print("We made it to the eval loop")
        for i in range(len(population.problem_descriptions)):
            # print(f'iteration {i} for this prompt: {self.prompt}')
            problem_description = population.problem_descriptions[i]
            option = population.options[i]
            answer = population.answers[i]
            
            # Truncate each part to fit within the specified limits
            
            prompt_truncated = self.truncate_to_fit(self.prompt, max_tokens_prompt)
            #option_truncated = self.truncate_to_fit(option, max_tokens_option)
            
            # Recalculate the remaining tokens for description after truncation
            completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt_truncated)])
            prompt_token_count = len(self.tokenizer.encode_chat_completion(completion_request).tokens)
            #option_token_count = len(self.tokenizer.encode(option_truncated))
            max_tokens_description = total_max_tokens - prompt_token_count - max_tokens_option
            
            description_truncated = self.truncate_to_fit(problem_description, max_tokens_description)
            
            response = llm(f": INSTRUCTION: {prompt_truncated}\
                               Reddit post: {description_truncated}\
                               Options: {option}\
                               ANSWER:",
                   max_tokens = 1024, 
                   stop = ['INSTRUCTION:'],
                   # frequency_penalty=0.7,
                   echo = False,) 
        # Evaluate the current prompt and return a score
           
            x = response['choices'][0]['text']
            print("EVALUATION RESULT:", x)
            #normalized_answer = self.normalize_answer(answer)
            x = self.replace_large_numbers(x)
#             self.prompt_history.append(x)
            # Extract the answer
            label_pattern = re.compile(r"(Current Direct|Current Indirect|Past Direct|Past Indirect|General/Informational|Hypothetical/Future|News/Noise)", re.IGNORECASE)
            # label_pattern = re.compile(re.escape(answer), re.IGNORECASE)
            
            # Find all matching labels in the response
            matches = label_pattern.findall(x)
            if matches:
                 # If there are matching labels, use the last match
                res = matches[-1]
                # print("This is the extracted answer:", res)
            else:
                # print("No answer found in the text.")
                res = None
            self.answer_tracker.append((population.generation, i, res, answer))
            
            if res and res.lower() == answer.lower():
                print("THEY MATCHED")
                accuracy +=1
            else:
                print("THEY DID NOT MATCH. THIS IS THE ANSWER:", answer, 'AND THIS IS THE RESULT', res)
                
        return accuracy

    def calculate_distance(self, other_member):
        return 1 - cosine(self.location, other_member.location)
    
    def mutate(self, parent_prompt, parent_id, Population, mutation_attempts=None):
        # Initialize mutation_attempts if it's not provided
        if mutation_attempts is None:
            mutation_attempts = 0
    
        self.parent_id = parent_id
    
        # Choose mutation method here
        mutated_prompt = self._generate_mutated_prompt(Population)
        
        if mutated_prompt:
            if self.prompt_proofing(mutated_prompt, Population):
                self.prompt = mutated_prompt
            else:
                print('Mutation unsuccessful - proofing failed!')
                mutation_attempts += 1
                if mutation_attempts < 3:
                    self.mutate(parent_prompt, parent_id, Population, mutation_attempts)
        else:
            print('Mutation unsuccessful - mutation failed!')
            mutation_attempts += 1
            if mutation_attempts < 3:
                self.mutate(parent_prompt, parent_id, Population, mutation_attempts)
            
        self.prompt_history.append(mutated_prompt)  # Log the mutated prompt
#         update_promptimal_frontier(mutated_prompt)
        
        # Check if the prompt exceeds the maximum length
    def _generate_mutated_prompt(self, population):
        operator = np.random.randint(1,8)
        # operator = 7
        # print("The mutation operator was ", operator)
        if len(self.prompt)>8192:
            self.prompt = self.prompt[:8192]
        if operator == 1:
            x = self.zero_order_direct_mutation(population)
        elif operator == 2:            
            x = self.first_order_direct_mutation(population)
        elif operator == 3:
            x = self.EDA_mutation(population)
        elif operator == 4:
            x = self.EDA_rank_index_mutation(population)
        elif operator == 5:
            x = self.lineage_based_mutation(population)
        elif operator == 6:
            x = self.zero_order_hyper_mutation(population)
        elif operator == 7:
            x = self.first_order_hyper_mutation(population)
        # elif operator == 8:
        #     x = self.Lamarckian_working_out_mutation(population)

        return x

        # elif operator == 9:
        #     x = self.crossover_mutation(self, population)
        # elif operator == 10:
        #     x = self.context_shuffling_mutation(self, population)
        
        
    def zero_order_direct_mutation(self, population):
        response = llm(f":INSTRUCTION: {population.prefix} + A numbered list of 100 hints:",
                   max_tokens = 1024, 
                   # frequency_penalty=0.7,
                   stop = ['INSTRUCTION'],
                   echo = False)

        x = response['choices'][0]['text']
        # print(x)
        
        # Pattern to match a numbered item: 
        # - Starts with a number followed by a period and a space
        # - Numbers within parentheses, e.g., (1), (2), etc.
        # - Optionally supports other common list markers like +, -, *, and #
        # - Allows for different numbering formats such as a), 1), i), I)
        pattern = re.compile(r'^(\+|-|\*|#|\(\d+\)|\d+\.\s+|\d+\)\s+|[a-zA-Z]\)\s+|[ivxlcdmIVXLCDM]+\)\s+)(.*)', re.MULTILINE)
        
        # Find all matches, but we only want the first one
        matches = pattern.findall(x)
        
        # Extract the first match if available
        first_list_item = matches[0][1] if matches else None

        return first_list_item
        
    def first_order_direct_mutation(self, population):
        mp = np.random.choice(population.mutation_prompts)
        response = llm(f": INSTRUCTION: Create a mutated prompt that solves a problem given the following information.\
                    Mutation instructions: {mp} \
                    Problem to solve: {population.task}",
                   max_tokens = 1024, 
                   # frequency_penalty=0.7,
                   stop = ['INSTRUCTION'],
                   echo = False)

        x = response['choices'][0]['text']
        return x

    def EDA_mutation(self, population):
        filtered_indices = []

        # Iterate through each sample in the distance matrix
        z = len(population.members)
        for i in range(z):
            # Get the similarity scores of the current sample with all others
            similarities = population.distance_matrix[i]
            if isinstance(similarities, np.ndarray):
                similarities = torch.from_numpy(similarities)
            # Mask to exclude the similarity of the sample with itself
            mask = torch.ones_like(similarities, dtype=torch.bool)
            mask[i] = False
            
            # Check if all other samples have a similarity score less than 0.95 with the current sample
            if torch.all(similarities[mask] < 0.95):
                filtered_indices.append(i)
        
        # Assuming 'members' is your original list and 'filtered_indices' is the list of indices to exclude
        p_filtered = [member.prompt for i, member in enumerate(population.members) if i not in filtered_indices]
        if len(p_filtered)==0:
            x = self.zero_order_direct_mutation(population)
            return x

        # def truncate_list(prompts_sorted, max_tokens):
        #     total_tokens = 0
        #     truncated_list = []
        #     for prompt in prompts_sorted:
        #         prompt_tokens = len(prompt) // 4  # Assuming each token has 4 characters
        #         if total_tokens + prompt_tokens <= max_tokens:
        #             truncated_list.append(prompt)
        #             total_tokens += prompt_tokens
        #         else:
        #             break
        #     return truncated_list

        truncated_prompts = self.truncate_list(p_filtered, 2000)
        random.shuffle(p_filtered)
        response = llm(f": A continued list of new task-prompts: {truncated_prompts}",
                   max_tokens = 1024, 
                   # frequency_penalty = 0.7,
                   # stop = ['INSTRUCTION'],
                   echo = False)

        x = response['choices'][0]['text']
        pattern = re.compile(r'^(\+|-|\*|#|\(\d+\)|\d+\.\s+|\d+\)\s+|[a-zA-Z]\)\s+|[ivxlcdmIVXLCDM]+\)\s+)(.*)', re.MULTILINE)
        
        # Find all matches, but we only want the first one
        matches = pattern.findall(x)
        
        # Extract the first match if available
        first_list_item = matches[0][1] if matches else None

        return first_list_item

    def EDA_rank_index_mutation(self, population):
        filtered_indices = []
        z = len(population.members)
        # print('z', z)
        # Iterate through each sample in the distance matrix
        for i in range(z):
            # Get the similarity scores of the current sample with all others
            similarities = population.distance_matrix[i]
            if isinstance(similarities, np.ndarray):
                similarities = torch.from_numpy(similarities)
            # Mask to exclude the similarity of the sample with itself
            mask = torch.ones_like(similarities, dtype=torch.bool)
            mask[i] = False
            
            # Check if all other samples have a similarity score less than 0.95 with the current sample
            if torch.all(similarities[mask] < 0.95):
                filtered_indices.append(i)
        
        # Assuming 'members' is your original list and 'filtered_indices' is the list of indices to exclude
        p_filtered = [(member.prompt, member.score) for i, member in enumerate(population.members) if i not in filtered_indices]
        if len(p_filtered)==0:
            x = self.zero_order_direct_mutation(population)
            return x
        p_filtered_sorted = sorted(p_filtered, key=lambda x: x[1])  # Sort by member.score (the second element of each tuple)

        # Extract member.prompt from the sorted list
        prompts_sorted = [prompt for prompt, score in p_filtered_sorted]
        

        truncated_prompts = self.truncate_list(prompts_sorted, 1700)

        # print(f'prompt sorted, {len(prompts_sorted)}')
        mp = np.random.choice(population.mutation_prompts)
        mp = self.truncate_to_fit(mp, 250)
        response = llm(f":INSTRUCTION: Here is a list of prompts in descending order of score:{truncated_prompts}\
                        The last of these is the best prompt.\
                        Output a mutation of this best prompt according to these instructions: {mp}",
                   max_tokens = 1024, 
                   # frequency_penalty=0.7,
                   stop = ['INSTRUCTION'],
                   echo = False)

        x = response['choices'][0]['text']
        return x

    def lineage_based_mutation(self, population):
        lineage = self.best_ancestors
        lineage_sorted = sorted(lineage, key=lambda x: x[1])  # Sort by member.score (the second element of each tuple)

        # Extract member.prompt from the sorted list
        elites_sorted = [prompt for prompt, score in lineage_sorted]
        elites_sorted = self.truncate_list(elites_sorted, 1500)
        p = self.truncate_to_fit(self.prompt, 450)
        try:
            response = llm(f": INSTRUCTION:This is a list of a prompt's elite ancestors in ascending order of quality: {elites_sorted}\
                            And this is the prompt that descended from the above ancestors: {p}.\
                            This is the next prompt in this genetic line.",
                   max_tokens = 1024,
                   # frequency_penalty=0.7,
                   stop = ['INSTRUCTION'],
                   echo = False)

        except ValueError as e:
            # Handle cases where the token limit is exceeded
            print("Token limit exceeded, reducing prompt size")
            while True:
                try:
                    # Remove the last item from elites_sorted and try again
                    if elites_sorted:
                        elites_sorted.pop()
                    else:
                        print("Cannot reduce prompt further.")
                        break
                    response = llm(f": INSTRUCTION:This is a list of a prompt's elite ancestors in ascending order of quality: {elites_sorted}\
                            And this is the prompt that descended from the above ancestors: {p}.\
                            This is the next prompt in this genetic line.",
                   max_tokens = 1024,
                   # frequency_penalty=0.7,
                   stop = ['INSTRUCTION'],
                   echo = False)
                    break  # Break if successful
                except ValueError:
                    continue
        x = response['choices'][0]['text']
        return x    

    def zero_order_hyper_mutation(self, population):
        mp = np.random.choice(population.mutation_prompts)
        ts = np.random.choice(population.thinking_styles)
        mp_2 = llm(f": INSTRUCTION: Create a mutation of the following instructions for the given task.\
                    Thinking style: {ts} \
                    Problem to solve: {population.prefix}",
                   max_tokens = 1024,
                   # frequency_penalty=0.7,
                   stop = ['INSTRUCTION'],
                   echo = False)

        
        x = mp_2['choices'][0]['text']

        response = llm(f": INSTRUCTION: Mutate a prompt given the following information.\
                    Mutation instructions: {x}\
                    Task Description: {self.prompt}",
                   max_tokens = 1024, 
                   # frequency_penalty=0.7,
                   stop = ['INSTRUCTION'],
                   echo = False)
        y = response['choices'][0]['text']
        return y 

    def first_order_hyper_mutation(self, population):
        mp = np.random.choice(population.mutation_prompts)
        mp_2 = llm(f": INSTRUCTION: Summarize and improve the following prompt: {mp}",
                   max_tokens = 1024,
                   # frequency_penalty=0.7,
                   stop = ['INSTRUCTION'],
                   echo = False)

        
        x = mp_2['choices'][0]['text']
        
        response = llm(f": INSTRUCTION: Mutate a prompt given the following information.\
                    Mutation instructions: {x}\
                    Prompt to be mutated: {self.prompt}",
                   max_tokens = 1024, 
                   # frequency_penalty=0.7,
                   stop = ['INSTRUCTION'],
                   echo = False)
        y = response['choices'][0]['text']
        return y
        
def main():
    # dataset_file = 'fixed_dataset.pkl'
    # with open(dataset_file, 'rb') as f:
    #      dataset = pickle.load(f)
    
    # # Retrieve variables from the fixed dataset
    # task = dataset['task']
    # prefix = dataset['prefix']
    # examples = dataset['examples']
    # answers = dataset['answers']
    # Example usage
    f = open('LKD_experiences.json') 
    data = json.load(f) 
    f.close() 
    task = str(data['description'])
    prefix = str(data['task_prefix'])
    examples = []
    options = []
    answers = []
    pop_size = 50
    eval_size = 105

    target_counts = defaultdict(int)
    for example in data['examples']:
        target_counts[example['target']] += 1

    random.shuffle(data['examples'])
    selected_counts = defaultdict(int)
    max_per_target = eval_size // len(target_counts)

    for example in data['examples']:
       target = example['target']
       if selected_counts[target] < max_per_target:
           examples.append(example['input'])
           options.append(list(example['target_scores'].keys()))
           answers.append(target)
           selected_counts[target] += 1
       if len(examples) >= eval_size: 
           break
    # if len(examples) >0:
    #     print('we got some examples in', len(examples))
        
    max_generations = 20
    #populate = Population(num_prompts=pop_size, max_generations=max_generations, t_s=t_s, m_p=m_p, task=task, prefix=prefix, problem_descriptions=examples, answers=answers, options=options)
    #populate.run()
    print("about to open the old pickle!")
    with open('V2_coarse_grained_105_eval_20_gens_50_members_ducc_bias_17.pkl', 'rb') as file:
         populate = pickle.load(file)
         populate.run(current_gen=17)

if __name__ == "__main__":
    main()
