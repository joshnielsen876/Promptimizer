import random
import matplotlib.pyplot as plt


# def initialize_settings():
#     return {
#         "Instruction": ["Error Handling Instructions", "Step-by-Step Guidance", "Simulate Dialogue Until Reaching Consensus",
#                         "Provide Examples with Explanations", "Outline Common Pitfalls", "Offer Alternative Approaches",
#                         "Include Visual Aids", "Give Real-World Applications", "Use Analogies", "Clarify Jargon"],
#         "Context": ["General Context", "Detailed Context", "Background Information", "Specific Context", "User Information",
#                     "Environmental Context", "Scenario Description", "Historical Context", "Cultural Context", "Geographical Context"],
#         "Creativity": ["High Creativity", "Moderate Creativity", "Low Creativity", "No Creativity", "Creative Thinking",
#                        "Innovative Approach", "Artistic Flair", "Creative Problem Solving", "Lateral Thinking", "Creative Writing"],
#         "Detail Level": ["High Detail", "Moderate Detail", "Low Detail", "Simple", "Detailed"],
#         "Domain Knowledge": ["High Domain Knowledge", "Moderate Domain Knowledge", "Low Domain Knowledge"],
#         "Examples": ["Single Example", "Multiple Examples", "Examples with Explanations", 
#                      "Examples without Explanations", "Comparative Examples", "Contrasting Examples", 
#                      "Step-by-Step Examples", "Real-world Examples", "Hypothetical Examples", "Counter Examples"],
#         "Intended Audience": ["General Public", "Students", "Professionals", "Experts", "Children", "Teenagers", "Adults", "Seniors",
#                               "Academics", "Researchers"],
#         "Tone": ["Formal", "Casual", "Encouraging", "Neutral", "Urgent", 
#                  "Sympathetic", "Inquisitive", "Confident", "Humorous", "Inspirational"],
#         "Personas": ["Data Scientist", "Research Analyst", "Storyteller", 
#                      "Tech Enthusiast", "Skeptical Reviewer", "Motivational Speaker", 
#                      "Cheerful Friend", "Skeptical Analyst", "Creative Thinker", "Detailed Instructor"]
#     }

import json

def ensure_json_closure(json_string):
    """Ensure that the JSON string has matching braces and quotes, returning the fixed string if necessary."""
    # Basic validation for unmatched brackets
    if json_string.count('{') != json_string.count('}'):
        print(f"Unmatched braces detected in JSON string.")
    if json_string.count('[') != json_string.count(']'):
        print(f"Unmatched square brackets detected in JSON string.")
    if json_string.count('"') % 2 != 0:
        print(f"Unmatched quotes detected in JSON string.")

    return json_string

def parse_json_string(response):
    try:
        json_style_string = response['choices'][0]['message']['content']
        # Remove unnecessary characters
        json_string = json_style_string.replace('\n', '')
        json_string = json_string.replace('\r', '')
        json_string = json_string.replace('\t', '')
        json_string = json_string.replace('\\', '')
        
        # Ensure the JSON structure is valid
        json_string = ensure_json_closure(json_string)
        
        print('this is the json string before loads:', json_string)

        # Parse the JSON string
        parsed_json = json.loads(json_string, strict=False)
        return parsed_json

    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        print("Offending JSON string:", json_string)
        raise e

def stochastic_tournament_selection(members, num_parents, tournament_size):
    selected_parents = []
    for _ in range(num_parents):
        # Randomly sample members to form the tournament
        tournament = random.sample(members, tournament_size)
        # Calculate the total fitness of the tournament
        total_fitness = sum(member.score for member in tournament)
        if total_fitness == 0:
            print("Total fitness is zero, cannot perform selection.")
            continue
        # Pick a random value between 0 and the total fitness
        pick = random.uniform(0, total_fitness)
        current = 0
        # Accumulate fitness scores and select a member
        for member in tournament:
            current += member.score
            # print(f"Current fitness: {current}, Member score: {member.score}")
            if current > pick:
                selected_parents.append(member)
                # print(f"Selected member with score: {member.score}")
                break
    return selected_parents

def random_crossover(parent1, parent2):
    crossover_methods = [one_point_crossover, two_point_crossover, uniform_crossover]
    crossover_method = random.choice(crossover_methods)
    return crossover_method(parent1.chromosome, parent2.chromosome)

def one_point_crossover(parent1, parent2):
    min_length = min(len(parent1), len(parent2))
    point = random.randint(1, min_length - 1)
    child1 = parent1[:point] + parent2[point:]
    return child1

def two_point_crossover(parent1, parent2):
    min_length = min(len(parent1), len(parent2))
    point1 = random.randint(1, min_length - 2)
    point2 = random.randint(point1 + 1, min_length - 1)
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    return child1

def uniform_crossover(parent1, parent2):
    min_length = min(len(parent1), len(parent2))
    child1 = []
    for i in range(min_length):
        if random.random() < 0.5:
            child1.append(parent1[i])
        else:
            child1.append(parent2[i])
    if len(parent1) > min_length:
        child1.extend(parent1[min_length:])
    if len(parent2) > min_length:
        child1.extend(parent2[min_length:])
    return child1


def random_resetting_mutation(chromosome, mutation_rate, settings):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = random.randint(0, len(settings[list(settings.keys())[i]])-1)
    return chromosome

def elitism(members, elite_size):
    elite_indices = sorted(range(len(members)), key=lambda i: members[i].score, reverse=True)[:elite_size]
    return [members[i] for i in elite_indices]

def replace_population(old_members, new_members):
    combined_population = old_members + new_members
    combined_fitnesses = [member.score for member in combined_population]
    sorted_indices = sorted(range(len(combined_population)), key=lambda i: combined_fitnesses[i], reverse=True)
    new_population = [combined_population[i] for i in sorted_indices[:len(old_members)]]
    return new_population

def initialize_contribution_scores(settings):
    scores = {}
    for component in settings:
        scores[component] = [0] * len(settings[component])
    return scores

def update_contribution_scores(scores, members, fitnesses, settings):
    for i, member in enumerate(members):
        fitness = fitnesses[i]
        for j, component in enumerate(settings):
            scores[component][member.chromosome[j]] += fitness
    return scores

def normalize_scores(scores):
    normalized_scores = {}
    for component, score_list in scores.items():
        total = sum(score_list)
        if total > 0:
            normalized_scores[component] = [s / total for s in score_list]
        else:
            normalized_scores[component] = score_list
    return normalized_scores

def visualize_scores(scores):
    fig, axes = plt.subplots(len(scores), 1, figsize=(10, 2 * len(scores)))
    if len(scores) == 1:
        axes = [axes]
    
    for ax, (component, score_list) in zip(axes, scores.items()):
        ax.bar(range(len(score_list)), score_list, tick_label=settings[component])
        ax.set_title(f'Contribution Scores for {component}')
        ax.set_xlabel('Setting')
        ax.set_ylabel('Normalized Score')
        ax.set_xticklabels(settings[component], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

def check_termination(generation, max_generations):
    return generation >= max_generations
    
def normalize_distance_matrix(distance_matrix):
    min_distance = np.min(distance_matrix)
    max_distance = np.max(distance_matrix)
    normalized_matrix = (distance_matrix - min_distance) / (max_distance - min_distance)
    return normalized_matrix

def filter_by_score(members, min_score):
    return [member for member in members if member.score >= min_score]

def optimize_diversity(candidates, existing_frontier, distance_matrix, similarity_limit):
    selected = []
    while candidates:
        best_candidate = None
        best_candidate_score = -1
        
        for candidate in candidates:
            candidate_index = candidate.index
            distances = [distance_matrix[candidate_index, existing_frontier.index(frontier_member)] for frontier_member in existing_frontier]
            if selected:
                distances += [distance_matrix[candidate_index, selected.index(sel_member)] for sel_member in selected]
            avg_distance = np.mean(distances)
            
            if avg_distance > best_candidate_score:
                best_candidate_score = avg_distance
                best_candidate = candidate
        
        if best_candidate_score >= similarity_limit:
            selected.append(best_candidate)
            candidates.remove(best_candidate)
        else:
            break
    return selected
