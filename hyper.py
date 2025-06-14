import random
# import openai
from llm import get_llm_instance
import json
from utils import *
import numpy as np

# Initialize Contribution Scores
def initialize_contribution_scores(settings):
    setting_scores = {component: {setting: 0 for setting in settings[component]} for component in settings}
    component_scores = {component: 0 for component in settings}
    return setting_scores, component_scores
    
def calculate_contribution_scores(members, settings):
    setting_scores = {component: {setting: 0 for setting in settings[component]} for component in settings}
    component_scores = {component: 0 for component in settings}
    for member in members:
        assert len(member.chromosome) == len(settings.keys()), f'mismatched chromosome length and key length {len(member.chromosome), len(settings.keys())}'
        for component, setting_index in zip(settings.keys(), member.chromosome):
            try:
                setting = settings[component][setting_index]
                setting_scores[component][setting] += member.score
                component_scores[component] += member.score
            except IndexError as e:
                print(e)
                print('component, setting index, settings.keys, member.chromosome', component, setting_index, settings.keys(), member.chromosome)
    
    # Normalize scores
    for component in setting_scores:
        max_score = max(setting_scores[component].values(), default=1)
        for setting in setting_scores[component]:
            if max_score !=0:
                setting_scores[component][setting] /= max_score
            # else:
            #     setting_scores[component][setting] /= 1
    
    # print(component_scores.values())
    max_component_score = max(component_scores.values(), default=1)
    if max_component_score == 0:
        max_component_score = 1
    print(max_component_score)
    for component in component_scores:
        # print(component_scores[component], max_component_score)
        # if component_scores[component]==0:
        #     component_scores[component]=1
        component_scores[component] /= max_component_score
    # print("these are the setting scores", setting_scores)
    # print("these are the component scores", component_scores)
    return setting_scores, component_scores

# Calculate Component Contribution Scores
def calculate_component_scores(scores):
    component_scores = {}
    for component, score_list in scores.items():
        component_scores[component] = sum(score_list)
    return component_scores

# Update Contribution Scores
def update_contribution_scores(scores, population, fitnesses, settings):
    for i, member in enumerate(population):
        fitness = fitnesses[i]
        for j, component in enumerate(settings):
            scores[component][member.chromosome[j]] += fitness
    return scores

# Normalize Scores
def normalize_scores(scores):
    normalized_scores = {}
    for component, score_list in scores.items():
        total = sum(score_list)
        if total > 0:
            normalized_scores[component] = [s / total for s in score_list]
        else:
            normalized_scores[component] = score_list
    return normalized_scores

# Identify Best and Worst Performing Settings
# def identify_performers(normalized_scores, settings, k=1):
#     best_performers = {}
#     worst_performers = {}
#     for component, score_list in normalized_scores.items():
#         sorted_indices = sorted(range(len(score_list)), key=lambda x: score_list[x])
#         worst_performers[component] = sorted_indices[:k]
#         best_performers[component] = sorted_indices[-k:]
#     return best_performers, worst_performers


def add_new_settings_and_components(settings, components, setting_scores, component_scores, task):
    # Add new settings to each component
    for component in settings:
        new_setting = generate_new_setting(component, settings)
        settings[component].append(new_setting)
    
    # Optionally add new components based on their scores
    best_component = max(component_scores, key=component_scores.get)
    new_component, new_settings = generate_new_component(settings, best_component, task)
    settings[new_component] = new_settings
    
    return settings

def ensure_json_closure(json_string):
    # Ensure the JSON string is properly closed with necessary characters
    open_braces = json_string.count('{')
    close_braces = json_string.count('}')
    if open_braces > close_braces:
        json_string += '}' * (open_braces - close_braces)
    return json_string
    
# Generate New Setting Using LLM
# We may need a few different prompts to imitate the hyper mutations from PromptBreeder
def generate_new_setting(component, settings, best_setting, worst_setting, task):
    current_settings = settings[component]
    llm_instance = get_llm_instance()
    response = llm_instance.create_chat_completion(messages=[
                            {'role':'System',
                             'content': f":You are an expert in writing and designing instructions for AI language models.\
                                We say that instructions/prompts are composed of components, each of which have different settings.\
                                You will be helping to find the optimal settings based on given information, returning your answer in JSON format."},
                            {'role': 'user',
                             'content':f"Use your expertise to help us make a better prompt component setting for a task. This is the task to be performed: {task}.\
                                The prompt component in question is {component}.\
                                The following are the current settings for the {component} component of a prompt: {current_settings}. \
                                The best setting is {best_setting} and the worst is {worst_setting}.\
                                Suggest a new setting that will be better than all the others, with your final answer in the required JSON format."}
            ],
           response_format={
            "type":'json_object',
            "schema": {
                "type":"object",
                "properties": {"Response and reasoning": {"type":"string"}, "New Setting" : {"type":"string"}},
                "required": ["New Setting"]}
            },
           max_tokens = 1000,
           temperature=0.0)

    json_style_string = response['choices'][0]['message']['content']
    json_string = json_style_string.replace('\n', '')
    json_string = json_string.replace('\r', '')
    json_string = json_string.replace('\t', '')
    json_string = json_string.replace('\\', '')   
    # Parse the JSON string
    try:
        parsed_json = json.loads(json_style_string, strict=False)
    except json.JSONDecodeError as e:
        print(f'Error:{e}')
        ensured_json_string = ensure_json_closure(json_style_string)
        try:
            json_string = ensured_json_string.replace('\n', '')
            json_string = json_string.replace('\r', '')
            json_string = json_string.replace('\t', '')
            json_string = json_string.replace('\\', '')   
            parsed_json = json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f'Second Error: {e}')     
            print(json_string)
    
    # Extract the generated text portion
    try:
        generated_text = parsed_json["New Setting"]
        print()
        # print('successfully found a SETTING on first attempt')
    except UnboundLocalError as e:
        print(f'Third Error: {e}')
        generated_text = json_string
    return generated_text


def generate_new_component(settings, best_component, task):
    # current_settings = settings[component]
    # worst_setting = settings[worst_setting_index]
    llm_instance = get_llm_instance()
    response = llm_instance.create_chat_completion(messages=[
                            {'role':'System',
                             'content': f":You are an expert in writing and designing instructional prompts for AI language models.\
                                We say that instructions/prompts are composed of components, each of which have different settings.\
                                You will be helping to find and create the optimal components and settings based on given information, returning your answer in JSON format."},
                            {'role': 'user',
                             'content':f"Use your expertise to help us make a better component of an instructional prompt for completing a task. This is the task to be performed: {task}.\
                                 The current prompt components and their settings are given as: {settings}.\
                                 Currently, the best performing component is {best_component}.\
                                 Considering the task, suggest a new component to be added to our list of settings, with a suitable name that describes the settings it can take.\
                                 The FIRST setting of the new component MUST be 'N/A', as in the given settings. Then add up to 5 additional settings that the component could take.\
                                 Give your answer in the required JSON format."}
            ],
           response_format={
            "type":'json_object',
            "schema": {
                "type":"object",
                "properties": {"Response and reasoning": {"type":"string"}, "New Component Name" : {"type":"string"}, "New Component Settings": {"type":"list"}},
                "required": ["New Component Name", "New Component Settings"]}
            },
           max_tokens = 1000,
           temperature=0.0)

    json_style_string = response['choices'][0]['message']['content']
    json_string = json_style_string.replace('\n', '')
    json_string = json_string.replace('\r', '')
    json_string = json_string.replace('\t', '')
    json_string = json_string.replace('\\', '') 
    # print("json output after adding new component", json_string)  

    # Parse the JSON string
    try:
        parsed_json = json.loads(json_string, strict=False)
        print()
        # print('successefully found a COMPONENT on first attempt')
    except json.JSONDecodeError as e:
        print(f'Error:{e}')
        ensured_json_string = ensure_json_closure(json_style_string)
        try:
            json_string = ensured_json_string.replace('\n', '')
            json_string = json_string.replace('\r', '')
            json_string = json_string.replace('\t', '')
            json_string = json_string.replace('\\', '')   
            parsed_json = json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f'Second Error: {e}. This is the string:{json_string}')
    # Extract the generated text portion
    if "New Component Name" in parsed_json and "New Component Settings" in parsed_json:
        # Expected format
        name = parsed_json["New Component Name"]
        setgs = parsed_json["New Component Settings"]
    else:
        name, setgs = next(iter(parsed_json.items()))
    return name, setgs
    

# Function to add new settings
def add_new_settings(settings, components, setting_scores, task, max_settings_size):
    shuffled_components = components[:]
    random.shuffle(shuffled_components)
    
    for component in shuffled_components:
        if len(settings[component])>=max_settings_size:
            print(f"{component} has enough settings for now. Skipping this mutation")
            continue
        elif np.random.random()>0.5:
            if component not in setting_scores:
                continue
            best_setting = max(setting_scores[component], key=setting_scores[component].get)
            worst_setting = min(setting_scores[component], key = setting_scores[component].get)
            new_setting = generate_new_setting(component, settings, best_setting, worst_setting, task)
            settings[component].append(new_setting)
            setting_scores[component][new_setting]=0
    # print()
    # print('after the add_new_settings function, 
    return settings, setting_scores

# Function to add new components
def add_new_component(settings, setting_scores, component_scores, task):
    best_component = max(component_scores, key=component_scores.get)
    new_component, new_settings = generate_new_component(settings, best_component, task)
    
    if new_component not in settings:
        settings[new_component] = new_settings
        component_scores[new_component] = 0
        setting_scores[new_component] = {setting: 0 for setting in new_settings}
        print()
        print('Added a new component. This is the new component and all updated components', new_component, settings.keys())
    else:
        print('tried to add a component but the component is already in settings')
    return settings, component_scores, setting_scores

def prompt_feedback(prompt, tuples, task):
    # current_settings = settings[component]
    # worst_setting = settings[worst_setting_index]
    llm_instance = get_llm_instance()
    # print(tuples)
    response = llm_instance.create_chat_completion(messages=[
                            {'role':'System',
                            'content': f"You are an expert in analyzing instructional prompts for AI language models.\
                                You will be helping to find and create optimal prompts based on given information, returning your answer in JSON format."},
                            {'role': 'user',
                             'content': f"Use your expertise to help us make a better instructional prompt. This is the task to be performed: {task}.\
                                 The prompt we are evaluating is {prompt}.\
                                 Here is a list of tuples of the reddit posts the prompt was used to classify, along with the generated answer, and the correct answer.\
                                 The format is (post, generated answer, correct answer, match True/False): {tuples}\
                                 First, examine why the prompt got some classifications right (if any) or wrong (if any).\
                                 Based on your examination, please generate an improved version of the prompt that would correctly classify all the samples.\
                                 Give your answer in the required JSON format."}
            ],
           response_format={
            "type":'json_object',
            "schema": {
                "type":"object",
                "properties": {"Examination": {"type":"string"}, "New Prompt" : {"type":"string"}},
                "required": ["New Prompt"]}
            },
           max_tokens = 1000,
           temperature=0.0)

    json_style_string = response['choices'][0]['message']['content']
    # print(json_style_string)
    json_string = json_style_string.replace('\n', '')
    json_string = json_string.replace('\r', '')
    json_string = json_string.replace('\t', '')
    json_string = json_string.replace('\\', '')   

    # Parse the JSON string
    try:
        parsed_json = json.loads(json_style_string, strict=False)
    except json.JSONDecodeError as e:
        print(f'Error:{e}')
        ensured_json_string = ensure_json_closure(json_style_string)
        try:
            json_string = ensured_json_string.replace('\n', '')
            json_string = json_string.replace('\r', '')
            json_string = json_string.replace('\t', '')
            json_string = json_string.replace('\\', '')   
            parsed_json = json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f'Second Error: {e}')   
            print(json_string)
    
    
    # Extract the generated text portion
    generated_text = parsed_json["New Prompt"]
    # print()
    # print("this is the new prompt based on feedback", generated_text)
    # print()
    return generated_text




def get_settings_from_prompt(prompt, settings):
    # current_settings = settings[component]
    # worst_setting = settings[worst_setting_index]
    llm_instance = get_llm_instance()
    response = llm_instance.create_chat_completion(messages=[
                            {'role':'System',
                            'content': f"You are an expert in analyzing and creating instructional prompts for AI language models.\
                                We say that instructions/prompts are composed of components, each of which have different settings.\
                                A prompt can be represented as a list of integers which serve as indices to the specific settings for a component.\
                                These lists allow us to treat a prompt as a chromosome for evolution in a genetic algorithm.\
                                You will be helping to transform prompts into this representation, returning your answer with a JSON schema. You MUST use the key 'Used Settings'."},
                            {'role': 'user',
                             'content': f"The following prompt was created to accomplish a task: {prompt}.\
                                 We need to convert this prompt into a list of integers which serve as indices to prompt settings, like a chromosome in a genetic algorithm.\
                                 Here are the settings we currently have: {settings}. \
                                 If no setting for a given component is used, that is 'N/A'. IF AND ONLY IF you belive that the prompt uses a setting which is not contained in\
                                 the above dictionary, report that as a string (dict value) that will be later added to the above settings.\
                                 IF AND ONLY IF a new component is also necessary, report that new component as a string (dict key) that will be later added to the above settings.\
                                 Identify which of these settings are used in this prompt and report it as a list of strings (not integers).\
                                 Give your answer in the required JSON format."}
            ],
           response_format={
            "type":'json_object',
            "schema": {
                "type":"object",
                "properties": {"Examination": {"type":"string"}, "Used Settings" : {"type":"list"}},#, "New Setting": {'type':'string'}, 'New Component': {'type':'string'}},
                "required": ["Used Settings"]}
            },
           max_tokens = 1000,
           temperature=0.0)

    json_style_string = response['choices'][0]['message']['content']
    json_string = json_style_string.replace('\n', '')
    json_string = json_string.replace('\r', '')
    json_string = json_string.replace('\t', '')
    json_string = json_string.replace('\\', '')   
    print('prompt to settings output', json_string)
    # Parse the JSON string
    try:
        parsed_json = json.loads(json_string, strict=False)
    except json.JSONDecodeError as e:
        print(f'Error:{e}')
        ensured_json_string = ensure_json_closure(json_style_string)
        try:
            json_string = ensured_json_string.replace('\n', '')
            json_string = json_string.replace('\r', '')
            json_string = json_string.replace('\t', '')
            json_string = json_string.replace('\\', '')   
            parsed_json = json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f'Second Error: {e}')   
            print(json_string)
    if 'Used Settings' in parsed_json:
        generated_text = parsed_json["Used Settings"]
        print()
        print("these are the identified settings from feedback, found expected key", generated_text)
        print()
    else:
        generated_text = parsed_json
        try:
            print("these are the identified settings from feedback, did NOT find expected key", [(key, value_list) for key, value_list in generated_text.items()])
        except TypeError as e:
            print(e, 'But it seems to be working anyway?')
            print("this is the generated text", generated_text)
    return generated_text

def settings_to_chromosome(generated_text, settings):
    indices_list = []
    
    # Validate that `generated_text` is a dictionary
    if not isinstance(generated_text, dict):
        raise ValueError("Expected generated_text to be a dictionary")
    
    for key, value_list in generated_text.items():
        if isinstance(value_list, list):  # If value_list is a list
            for value in value_list:
                if key in settings:
                    if value not in settings[key]:
                        settings[key].append(value)
                    index = settings[key].index(value)
                else:
                    settings[key] = [value]
                    index = 0
                indices_list.append(index)
        elif isinstance(value_list, str):  # If value_list is a string
            value = value_list  # Treat the string as a single value
            if key in settings:
                if value not in settings[key]:
                    settings[key].append(value)
                index = settings[key].index(value)
            else:
                settings[key] = [value]
                index = 0
            indices_list.append(index)
        else:  # Handle unexpected types gracefully
            print(f"Warning: The value for {key} is of unexpected type {type(value_list)} and will be skipped.")
    
        return indices_list, settings

    