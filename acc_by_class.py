import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np


def analyze_results_by_generation(answer_tracker):
    generation_data = defaultdict(list)
    
    # Group data by generation
    for generation, item_index, result, response, answer in answer_tracker:
        result = result.lower() if result is not None else None
        answer = answer.lower()
        generation_data[generation].append((item_index, result, answer))
    
    generation_accuracies = {}
    generation_none_counts = defaultdict(int)
    
    # Calculate accuracy and None counts for each generation
    for generation, data in generation_data.items():
        total_items = len(data)
        correct_items = 0
        
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        for _, result, answer in data:
            class_total[answer] += 1
            if result == answer:
                correct_items += 1
                class_correct[answer] += 1
            if result is None:
                generation_none_counts[generation] += 1
        overall_score = correct_items  # Out of 30
        class_scores = {cls: class_correct[cls] for cls in class_total}
        # class_scores = {cls: class_correct[cls] for cls in class_total}
        
        generation_accuracies[generation] = (overall_score, class_scores)
    
    return generation_accuracies, generation_none_counts



def aggregate_accuracies(members, use_classes=False):
    if use_classes:
        # Original class-based aggregation logic
        aggregated_overall_scores = defaultdict(list)
        aggregated_class_scores = defaultdict(lambda: defaultdict(list))
        aggregated_none_counts = defaultdict(list)
        
        for member in members:
            generation_accuracies, generation_none_counts = analyze_results_by_generation(member.answer_tracker)
            
            for generation, (overall_score, class_scores) in generation_accuracies.items():
                aggregated_overall_scores[generation].append(overall_score)
                for cls, score in class_scores.items():
                    aggregated_class_scores[generation][cls].append(score)
            for generation, none_count in generation_none_counts.items():
                aggregated_none_counts[generation].append(none_count)
        
        return aggregated_overall_scores, aggregated_class_scores, aggregated_none_counts
    else:
        # Simplified aggregation without class distinction
        generations = range(len(members[0].accuracy_history))  # Number of generations
        aggregated_overall_scores = {gen: [member.accuracy_history[gen] for member in members] for gen in generations}
        return aggregated_overall_scores, None, None  # No class scores or none counts


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

def Plotting(pop, use_classes):
    # Aggregate data based on the `use_classes` flag
    aggregated_overall_scores, aggregated_class_accuracies, _ = aggregate_accuracies(pop.members, use_classes=use_classes)
    
    # Prepare data for plotting
    generations = sorted(aggregated_overall_scores.keys())

    if use_classes and aggregated_class_accuracies:
        # Class-based plotting
        class_labels = sorted(aggregated_class_accuracies[generations[0]].keys())
        colors = plt.cm.get_cmap('tab10', len(class_labels))
        
        # Violin plots for class scores
        plt.figure(figsize=(14, 8))
        box_width = 0.6 / len(class_labels)
        legend_patches = []  # Store legend handles

        for i, cls in enumerate(class_labels):
            class_data = [aggregated_class_accuracies[g][cls] for g in generations]
            positions = [gen + (i - len(class_labels) / 2) * box_width for gen in generations]
            
            violin_parts = plt.violinplot(class_data, positions=positions, widths=box_width, showmeans=False, showmedians=True)
            
            # Apply colors properly
            for part in ('bodies', 'cbars', 'cmins', 'cmaxes', 'cmedians'):
                if part in violin_parts:
                    if part == 'bodies':
                        for pc in violin_parts[part]:
                            pc.set_facecolor(colors(i))
                            pc.set_edgecolor('black')
                            pc.set_alpha(0.7)
                    else:
                        violin_parts[part].set_color(colors(i))
            
            # Create a legend entry
            legend_patches.append(mpatches.Patch(color=colors(i), label=cls))

        # Add legend
        plt.legend(handles=legend_patches, title="Classes", loc="upper right")

        plt.xlabel('Generation')
        plt.ylabel('Accuracy')
        plt.title('Class-wise Accuracy Over Generations')
        plt.grid(True)
        plt.savefig('Results/LKD/3 class_V3 accuracy breakdown.png', dpi=600, bbox_inches='tight')
        plt.show()
        
        
        # Plot standard deviation of accuracy
        # plt.figure(figsize=(14, 8))
        # for i, cls in enumerate(class_labels):
        #     plt.plot(generations, std_accuracies[cls], marker='o', color=colors(i), label=f'{cls}')
        
        # plt.xlabel('Generation')
        # plt.ylabel('Standard Deviation of Accuracy (%)')
        # plt.title('Standard Deviation of Accuracy by Class and Generation')
        # plt.legend()
        # plt.grid(True)
        # plt.xticks(ticks=generations, labels=[int(gen) for gen in generations])
        # plt.savefig('Results/Standard Deviation of Accuracy by class and generation_exp_1.png', bbox_inches='tight')
        # plt.show()
        
    else:
        # Non-class-based plotting with mean accuracy line
        colors = plt.cm.viridis(np.linspace(0, 1, len(generations)))  # Color gradient for generations
        mean_accuracies = [np.mean(aggregated_overall_scores[gen]) for gen in generations]
        
        plt.figure(figsize=(14, 8))
        
        # Plot each generation's overall accuracy as a box plot
        for i, gen in enumerate(generations):
            data = aggregated_overall_scores[gen]
            vp = plt.violinplot(data, positions=[gen], widths=0.6, showmeans=False, showmedians=True)
            
            # Set the color for each violin plot
            for body in vp['bodies']:
                body.set_facecolor(colors[i])
                body.set_edgecolor(colors[i])
                body.set_alpha(0.7)
            vp['cmedians'].set_color('black')  # Set color of median line
        
        # Overlay mean accuracy line
        plt.plot(generations, mean_accuracies, marker='o', color='blue', linestyle='-', linewidth=2, label='Mean Accuracy')
        
        plt.xlabel('Generation')
        plt.ylabel('Accuracy (# correct out of 50)')
        # plt.title('Overall Accuracy Distribution Across Generations with Mean Accuracy Line')
        plt.legend()
        plt.grid(True)
        plt.xticks(ticks=generations, labels=generations)
        plt.savefig(f'Results/LKD/3_classes_v3.png')#Results/Experiment {experiment_num+1}/Accuracy by generation_exp_{experiment_num+1}_V4.png', dpi=600, bbox_inches='tight')
        plt.show()
        
        