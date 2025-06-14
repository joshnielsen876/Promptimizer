import memberLKD
from memberLKD import Member
from utils import *
from llm import get_llm_instance
import json
from sentence_transformers import SentenceTransformer, util
import numpy as np
from hyper import *
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine


class Population:
    def __init__(self, num_prompts, max_generations, settings, task, prefix, task_type, problem_descriptions, options, answers,**kwargs):
        self.settings = settings
        self.generation = 0
        self.max_generations = max_generations
        self.best_members = []
        self.global_best_prompt = []
        self.best_DFR_history = []
        self.elite_history = []
        self.global_best_score = [0]
        self.similarity_limit = 0.90
        self.task = task
        self.prefix = prefix
        self.problem_descriptions = problem_descriptions
        self.task_type = task_type
        print(self.problem_descriptions)
        self.options = options
        self.answers = answers
        self.accuracy_threshold = 0.1
        self.members = [Member(self) for i in range(num_prompts)]
        self.promptimal_frontier_history = []
        self.promptimal_frontier = [member for member in self.members]
        # self.promptimal_frontier = self.update_promptimal_frontier() #self.members.copy
        self.distance_matrix = np.ones((len(self.members), len(self.members)))
        self.population_distances()
        self.setting_scores, self.component_scores = initialize_contribution_scores(settings)
        self.norm_dm = self.normalize_distance_matrix()
        self.contribution_score_history = []
        self.parents = None
        self.parent_history = []
        self.was_mutated = False
        
        
        
    def normalize_distance_matrix(self):
        distance_matrix = np.array(self.distance_matrix)
        min_distance = np.min(distance_matrix)
        max_distance = np.max(distance_matrix)
        self.norm_dm = (distance_matrix - min_distance) / (max_distance - min_distance)
        
    def gen_prompt(self, decoded_settings, task):
        llm_instance = get_llm_instance()
        response = llm_instance.create_chat_completion(messages = [
            {"role": "system",
             "content": f":You are an expert in writing and designing instructions for language models.\
                            We say that instructions/prompts for a task can be composed of components, each of which have different settings.\
                            This is our task: {task}.\
                            You will be asked to create a prompt that incorporates some given settings, which will be used by an AI language model. to complete the task.\
                            If one of the given settings is 'N/A', ignore that component completely. \
                            Remember, what you write is exactly what will be provided to the AI model directly.\
                            Clearly identify the new prompt for easy extraction. Output your response in the given JSON format."
            },
            {"role": "user",
             "content": f"We say that instructions/prompts for a task can be composed of components, each of which have different settings.\
                            Task: {task},\
                            Settings: {decoded_settings},\
                            Create instructions based on these settings (but do not explicity state or repeat them) for an AI language model to use when completing the task.\
                            The AI will take on two personas who will dialogue until they reach a consensus on the classification task.\
                            Think of it this way: if the instructions/prompt you created were analyzed, these features would be found, even if not explicitly written.\
                            Output your response in JSON format with two fields: 'Response and reasoning', and 'New Prompt'."
            }],
            response_format={
            "type":'json_object',
            "schema": {
                "type":"object",
                "properties": {"Response and reasoning": {"type":"string"}, "New Prompt" : {"type":"string"}},
                "required": ["New Prompt"]}
            },
            max_tokens = 1800, 
            temperature=0.0)
            # echo = False)
        # print('This is the new response format', response)
        json_style_string = response['choices'][0]['message']['content']
        json_string = json_style_string.replace('\n', '')
        json_string = json_string.replace('\r', '')
        json_string = json_string.replace('\t', '')
        # json_string = json_string.replace('\\', '')        
        # Parse the JSON string
        try:
            parsed_json = json.loads(json_string, strict=False)
        except json.JSONDecodeError as e:
            print('this is the json string before loads that threw a decode error. We will try again.', json_string)
            r = llm_instance.create_chat_completion(messages = [
                                                    {"role":"System",
                                                     "content": "You are an expert in text processing, specifically preparing strings for the json.loads function."},
                                                    {"role":"user",
                                                     "content": f"The following json string returned the given error when json.loads was called.\
                                                                 String: {json_string}.\
                                                                 Error: {e}.\
                                                                 Please return the correctly formatted string so json.loads will work when it receives the string."}],
                                                   temperature=0)
            json_string = reponse['choices'][0]['message']['content']
            print("this is the updated json string after the LM was instructed to fix it", json_string)
            try:
                parsed_json = json.loads(json_string, strict=False)
            except json.JSONDecodeError as e:
                print("it still did not work, so we start again with a new LM call.")
                self.gen_prompt(decoded_settings, self.task)
        # Extract the generated text portion
        generated_text = parsed_json["New Prompt"]
        # print(generated_text)
        return generated_text

    #UPDATE THIS BASED ON DFR
    def evaluate_members(self):
        best_member = None
        best_member_score = 0
        # self.get_DFR()
        for member in self.members:
            if member.was_mutated:
                member.score = member.evaluate_prompt(self)
            else:
                new_generation = self.generation
                # print("Slicing answer tracker:", member.answer_tracker[-len(self.answers):])
                
                new_answers = [(new_generation, index, choice, response, answer) 
                       for _, index, choice, response, answer in member.answer_tracker[-len(self.answers):]]
                member.answer_tracker.extend(new_answers)
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

    def population_distances(self):
        model = SentenceTransformer("all-MiniLM-L6-v2").to('cpu')
        sentences = [a.prompt for a in self.members]
        embeddings = model.encode(sentences)
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
        fitness = np.array([member.score for member in self.members])/100
        normalized_distance = ((self.distance_matrix.numpy() if hasattr(self.distance_matrix, 'numpy') else self.distance_matrix) + 1) / 2
        np.fill_diagonal(normalized_distance, 0) 
        average_distance = normalized_distance.mean(axis=1)
        weight_fitness, weight_diversity = self.adjust_weights()
        DFR = (weight_fitness * fitness) + (weight_diversity * average_distance)
        for i, member in enumerate(self.members):
            member.DFR = DFR[i]
            member.DFR_history.append(DFR[i])
            member.avg_distance = average_distance[i]
            member.avg_distance_history.append(average_distance[i])
        
    
    def update_promptimal_frontier(self):
        self.population_distances()
        self.normalize_distance_matrix()
        
        current_max_score = max(member.score for member in self.members)
        # print([i.prompt for i in self.promptimal_frontier])
        frontier_max_score = max((member.score for member in self.promptimal_frontier), default=0)  # default=0 handles empty frontier
        max_score = max(current_max_score, frontier_max_score)
        min_score = max_score * (1 - self.accuracy_threshold)

        self.promptimal_frontier = [member for member in self.promptimal_frontier if member.score >= min_score]
        new_candidates = [member for member in self.promptimal_frontier if member.score >= min_score]
        
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
                
    def chromosome_from_feedback(self, member, attempts = None):
        p = member.prompt
        tup = member.select_unique_pairs()
        new_prompt = prompt_feedback(p, tup, self.task)
        
        # Debugging the output
        extracted = get_settings_from_prompt(new_prompt, self.settings)
        print(f"Extracted settings: {extracted}")
        
        # Process the settings into a chromosome
        try:
            new_chromo, self.settings = settings_to_chromosome(extracted, self.settings)
        except ValueError as e:
            print(f"Error converting settings to chromosome: {e}")
            raise
        
        return new_chromo, new_prompt

    
    def update_chromosomes_for_new_component(self, settings):
        for member in self.members:
            while len(member.chromosome)<len(settings):
                member.chromosome.append(0)
            assert len(member.chromosome) == len(settings), "After expanding chromosome, it's still not long enough, meaning that it needed more than one additional slot"
            
    def mutate(self, member, setting_scores, component_scores, max_components, max_settings_size, limited = False):
        if len(self.settings)>= max_components:
            print(f"There are enough components. No more mutations of this kind")
            mutation_type = "Add setting"
        else:
            if limited == False:
                mutation_type = np.random.choice(['Add setting', 'Add component', 'Feedback'])
            else:
                mutation_type = np.random.choice(['Add setting', 'Add component'])
        components = list(self.settings.keys())
        
        if mutation_type == 'Add setting':
            # self.setting_scores, self.component_scores = calculate_contribution_scores(self.members, self.settings)
            self.settings, self.setting_scores  = add_new_settings(self.settings, components, setting_scores, self.task, max_settings_size)
            new_member = Member(self)
            # print()
            print(f'added a setting to some components')#. There are now {len(self.settings)} components.')
            # print()
            new_chromo, new_prompt = new_member.chromosome, new_member.prompt
             
        elif mutation_type == 'Add component':
            self.settings, self.component_scores, self.setting_scores, = add_new_component(self.settings, setting_scores, component_scores, self.task)
            # print()
            print(f'added a component. There are now {len(self.settings)} components.')
            # print()
            new_member = Member(self)
            new_chromo, new_prompt = new_member.chromosome, new_member.prompt
            if len(new_chromo)>len(member.chromosome):
                self.update_chromosomes_for_new_component(self.settings)  # Update chromosomes
            
        elif mutation_type == 'Feedback':
            new_chromo, new_prompt = self.chromosome_from_feedback(member)
            attempts = 1
            while len(new_chromo) != len(self.settings):
                if attempts >3:
                    # new_chromo, new_prompt = member.chromosome, member.prompt
                    new_chromo, new_prompt, mutation_type = self.mutate(member, setting_scores, component_scores, max_components, max_settings_size, limited=True)
                else:
                    print("Failed feedback mutation, trying attempt number", attempts+1)
                    new_chromo, new_prompt = self.chromosome_from_feedback(member, attempts = attempts)
                    attempts +=1
                
            print('chromosome and prompt after feedback', new_chromo, new_prompt)
            if len(new_chromo)>len(member.chromosome):
                print()
                print(f'added a new component from feedback. There are now {len(self.settings)} components.')
                self.update_chromosomes_for_new_component(self.settings)
            
        # if len(new_chromo) != len(self.settings):
            
        return new_chromo, new_prompt, mutation_type
    
    def run(self, current_gen = None, max_components = None, max_settings_size = None, mutation_rate = None, experiment_num = None):
        if current_gen is not None:
            a, b = current_gen+1, self.max_generations
        else:
            a, b = 0, self.max_generations

        size = len(self.members)
        # mutation_rate = 0.1
        components_full = len(self.settings) >= max_components
        settings_full = all(len(self.settings[x]) >= max_settings_size for x in self.settings)
        for _ in range(a, b):
            print(f"START OF GENERATION {_}")
            self.generation = _
            if _ != 0:
                print("STARTING TO EVALUATE MEMBERS IN GENERATION", _)
                self.evaluate_members()
                print("FINISHED EVALUATING MEMBERS IN GENERATION", _)
            self.update_promptimal_frontier()
            self.best_members.append((self.generation, self.global_best_prompt, self.global_best_score))
            elites = elitism(self.members, elite_size=2)
            # new_population = elites.copy()
            setting_scores, component_scores = calculate_contribution_scores(self.members, self.settings)
            self.contribution_score_history.append((self.generation, self.setting_scores, self.component_scores))
            
            
            # Selection and crossover
            num_parents = (len(self.members)-len(elites))//2
            self.parents = stochastic_tournament_selection(self.members, num_parents, tournament_size=2)
            for i in elites:
                if i not in self.parents:
                    self.parents.append(i)
            self.parent_history.append(self.parents)
            for member in self.members:
                if member not in self.parents:
                    parent1, parent2 = random.sample(self.parents, 2)
                    child_chromosome = random_crossover(parent1, parent2)
                    #genetic mutation, different from hyper mutations
                    child_chromosome = random_resetting_mutation(child_chromosome, mutation_rate=0.01, settings=self.settings)
        
                    member.chromosome_history.append(member.chromosome)
                    member.chromosome = child_chromosome
                    decoded_settings = {component: self.settings[component][index] for component, index in zip(self.settings.keys(), member.chromosome)}
                    member.prompt = self.gen_prompt(decoded_settings, self.task)
                    member.prompt_history.append(member.prompt)
                    member.was_mutated = True
                    member.mutation_history.append("Child")
                else:
                    member.chromosome_history.append(member.chromosome)
                    member.prompt_history.append(member.prompt)
                    member.was_mutated = False
                    member.mutation_history.append(False)
            
            #hyper mutation changes after reproduction, two immigrant options and one feedback option      
            #Don't hyper mutate if our settings matrix is full
            count = 0
            if not components_full and not settings_full:
                for member in self.members:
                    if member not in elites:
                        print(f'considering hyper mutation of {member.chromosome} in generation {self.generation}')
                        if np.random.random() < mutation_rate:
                            print('hyper mutation activated')
                            member.chromosome, member.prompt, mutation_type = self.mutate(
                                member, setting_scores, component_scores, max_components, max_settings_size
                            )
                            count +=1
                            print("COUNT", count)
                            member.mutation_history.append(mutation_type)
                            member.was_mutated = True
                        else:
                            member.was_mutated = False
                            member.mutation_history.append(False)
                    else:
                        member.was_mutated = False
                        member.mutation_history.append(False)

            # Update flags after mutations
            components_full = len(self.settings) >= max_components
            settings_full = all(len(self.settings[x]) >= max_settings_size for x in self.settings)
                
            print("Here are the accuracies at the end of generation", self.generation, [i.score for i in self.members])
            # Use `with` statement to ensure proper resource management
            print("DUMPING PICKLE")
            print('\n'*3)
            mutation_rate_note = int(mutation_rate*100)
            # with open(f'Results/Experiment {experiment_num}/{mutation_rate_note}p_hyper_{max_components}C_{max_settings_size}S_generation{self.generation}_V3.pkl', 'wb') as file:
            with open(f'Results/LKD/3_classes_v3_generation{self.generation}.pkl', 'wb') as file:
                pickle.dump(self, file)
            print("PICKLE HAS BEEN DUMPED")