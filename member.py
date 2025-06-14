import uuid
import random
from utils import *
import json
import copy
from llm import get_llm_instance
from collections import defaultdict
import mistral_common
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    UserMessage,
    ToolMessage
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import re
from json_handling import *
import concurrent.futures


class Member:
    def __init__(self, population, chromosome=None, prompt=None, **kwargs):
        self.id = str(uuid.uuid4())
        self.answer_tracker = []
        self.tokenizer = MistralTokenizer.v1()
        if chromosome is not None and prompt is not None:
            self.chromosome = chromosome
            self.prompt = prompt
            self.score = 0
        else:
            self.chromosome, self.prompt = self.Initialize(population, chromosome_length=len(population.settings))
            self.score = self.evaluate_prompt(population)
            self.was_mutated = True
        self.chromosome_history = [self.chromosome]
        self.prompt_history = [self.prompt]
        self.accuracy_history = [self.score]
        self.best_prompt = copy.deepcopy(self.prompt)
        self.best_ancestors = []
        self.avg_distance = 0
        self.avg_distance_history = []
        self.DFR = 0
        self.DFR_history = []
        self.mutation_history = []
        
        
        
    def Initialize(self, population, chromosome_length):
        settings = population.settings
        chromosome = [random.randint(0, len(settings[component])-1) for component in settings]
        decoded_settings = {component: settings[component][index] for component, index in zip(settings.keys(), chromosome)}
        prompt = population.gen_prompt(decoded_settings, population.task)
        return chromosome, prompt    

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
            return tokenizer.decode(tokens) + '...'
        return text
        
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
        total_max_tokens = 2000
        max_tokens_prompt = 1800
        max_tokens_option = 50
        tokenizer = MistralTokenizer.v1()
        # print(len(population.problem_descriptions), len(population.answers))

        # def process_prompt(i):
        for i in range(len(population.problem_descriptions)):
            problem_description = population.problem_descriptions[i]
            answer = population.answers[i]

            if len(self.prompt) // 4 > 1800:
                prompt_truncated = self.truncate_to_fit(self.prompt, max_tokens_prompt)
                completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt_truncated)])
                prompt_token_count = len(tokenizer.encode_chat_completion(completion_request).tokens)
                max_tokens_description = total_max_tokens - prompt_token_count - max_tokens_option
                description_truncated = self.truncate_to_fit(problem_description, max_tokens_description)
            else:
                prompt_truncated = self.prompt
                description_truncated = problem_description

            llm_instance = get_llm_instance()
            response = llm_instance.create_chat_completion(messages=[
                   {"role": "system",
                         "content": "You are a helpful assistant that outputs in JSON. Given instruction and provided content, clearly output your integer answer AT THE END."
                    },
                  {"role": "user",
                   "content": f": INSTRUCTION: {prompt_truncated}\
                           Content: {description_truncated}"}
                   ],
                   response_format={
                        "type":'json_object',
                        "schema": {
                            "type": "object",
                            'properties': {"reasoning": {"type":"string"}, "Answer": {"type":"int"}},
                            "required": ["Answer"],
                       },
                   },
                   max_tokens = 1024,
                   temperature=0.0) 

            x = response['choices'][0]['message']['content']
            # x = self.replace_large_numbers(x)
            # Extract the answer
            label_pattern = re.compile(re.escape(answer), re.IGNORECASE)
            
            # Find all matching labels in the response
            matches = label_pattern.findall(x)
            if matches:
                res = matches[-1]
            else:
                res = None
            self.answer_tracker.append((population.generation, i, res, answer))
            
            if res and res.lower() == answer.lower():
                # print("THEY MATCHED")
                accuracy +=1
                # return 1
            # else:
                # print("THEY DID NOT MATCH")
                # print(f"The correct answer was {answer}, but we got {x}")
                # return 0
        
        # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        # results = list(executor.map(process_prompt, range(len(population.problem_descriptions))))
        print("FINISHED EVALUATING MEMBER", self.id)
        print("ACCURACY FOR THIS MEMBER WAS", accuracy)
        # accuracy = sum(results)
        return accuracy
        
#     def evaluate_prompt(self, population):
#         accuracy = 0
#         task = population.task
#         # Define maximum tokens for each part to ensure the total is within the limit
#         total_max_tokens = 2000

#         # Allocate tokens for each part, ensuring the total does not exceed the limit
#         max_tokens_prompt = 1800
#         max_tokens_option = 50
#         tokenizer = MistralTokenizer.v1()
#         for i in range(len(population.problem_descriptions)):
#             # print(f'iteration {i} for this prompt: {self.prompt}')
#             problem_description = population.problem_descriptions[i]
#             # option = population.options[i]
#             answer = population.answers[i]
            
#             if len(self.prompt)//4 > 1800:
#                 # Truncate each part to fit within the specified limits

#                 prompt_truncated = self.truncate_to_fit(self.prompt, max_tokens_prompt)
#                 # option_truncated = self.truncate_to_fit(option, max_tokens_option)

#                 # Recalculate the remaining tokens for description after truncation
#                 completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt_truncated)])
#                 prompt_token_count = len(tokenizer.encode_chat_completion(completion_request).tokens)
#                 # option_token_count = len(self.tokenizer.encode(option_truncated))
#                 max_tokens_description = total_max_tokens - prompt_token_count - max_tokens_option

#                 description_truncated = self.truncate_to_fit(problem_description, max_tokens_description)
#             else:
#                 prompt_truncated = self.prompt
#                 description_truncated = problem_description
                
#             llm_instance = get_llm_instance()
#             response = llm_instance.create_chat_completion(messages=[
#                    {"role": "system",
#                          "content": "You are a helpful assistant that outputs in JSON. Given instruction and provided content, select the correct option from those available.",
#                     },
#                   {"role": "user",
#                    "content": f": INSTRUCTION: {prompt_truncated}\
#                            Math problem: {description_truncated}"}
#                    ],
#                    response_format={
#                         "type":'json_object',
#                         "schema": {
#                             "type": "object",
#                             'properties': {"reasoning": {"type":"string"}, "Final Answer": {"type":"string"}},
#                             "required": ["Final Answer"],
#                        },
#                    },
#                    max_tokens = 1024) 
#                    # stop = ['INSTRUCTION:'],
#                    # frequency_penalty=0.7,
#                    # echo = False)
#             # Evaluate the current prompt and return a score
#             # print('This is the new output format', response)
#             json_style_string = response['choices'][0]['message']['content']
#             json_string = json_style_string.replace('\n', '')
#             json_string = json_string.replace('\r', '')
#             json_string = json_string.replace('\t', '')
#             json_string = json_string.replace('\\', '')
#             # Parse the JSON string
#             parsed_json = json.loads(json_style_string, strict=False)
            
#             # Extract the generated text portion
#             generated_text = parsed_json["Final Answer"]
#             # print("EVALUATION RESULT:", x)
# #             self.prompt_history.append(x)
#             # Extract the answer
#             label_pattern = re.compile(re.escape(answer), re.IGNORECASE)
#             # label_pattern = re.compile(r"(Current Direct|Current Indirect|Past Direct|Past Indirect|General/Informational|Hypothetical/Future|News/Noise)", re.IGNORECASE)
            
#             # Find all matching labels in the response
#             matches = label_pattern.findall(generated_text)
#             if matches:
#                  # If there are matching labels, use the last match
#                 res = matches[-1]
#                 # print("This is the extracted answer:", res)
#             else:
#                 # print("No answer found in the text.")
#                 res = None
#             self.answer_tracker.append((population.generation, i, res, answer))
#             # print('reddit content', description_truncated)
#             # print('result', res)
#             # print()
#             if res and res == answer:
#                 # print("THEY MATCHED")
#                 accuracy +=1
#             # else:
#                 # print("THEY DID NOT MATCH. THIS IS THE ANSWER:", answer, 'AND THIS IS THE RESULT', res)
                
#         return accuracy

    def calculate_distance(self, other_member):
        return 1 - cosine(self.location, other_member.location)

    def select_unique_pairs(self, num_pairs=3, ensure_unique=False):
        # Shuffle the answer_tracker list to randomize selection
        at = [i for i in self.answer_tracker]
        random.shuffle(at)
        
        selected_pairs = []
        
        if ensure_unique:
            # Use a dictionary to ensure unique answers
            unique_pairs = {}
            for entry in at:
                result, answer = entry[2], entry[3]
                if answer not in unique_pairs:
                    match = result == answer
                    unique_pairs[answer] = (result, answer, match)
                if len(unique_pairs) == num_pairs:
                    break
            
            # Convert the unique pairs dict to a list
            selected_pairs = list(unique_pairs.values())
            
            # Check if we have enough unique pairs
            if len(selected_pairs) < num_pairs:
                raise ValueError("Not enough unique answers to select the desired number of pairs.")
        
        else:
            # Select random pairs without checking for uniqueness
            for entry in at:
                result, answer = entry[2], entry[3]
                match = result == answer
                selected_pairs.append((result, answer, match))
                print('added a pair', (result, answer, match))
                
                if len(selected_pairs) == num_pairs:
                    break
            
            # Check if we have enough pairs
            if len(selected_pairs) < num_pairs:
                raise ValueError("Not enough answers to select the desired number of pairs.")
        
        return selected_pairs
