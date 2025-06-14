import matplotlib.pyplot as plt
import numpy as np

def plot_distances(pop):
    #accuracy over time
    acc = []
    dfrs = []
    count = 0
    for i in pop.members:
        acc.append(i.accuracy_history)
        dfrs.append(i.avg_distance_history)
        count+=1
    # Prepare data for plotting
    num_generations = len(dfrs[0])
    num_individuals = len(dfrs)
    
    # Transpose the DFRs list to group values by generation
    dfrs_by_generation = [[dfrs[individual][generation] for individual in range(num_individuals)] for generation in range(num_generations)]

    # Create the box and whisker plot with colors
    plt.figure(figsize=(14, 8))
    box = plt.boxplot(dfrs_by_generation, positions=range(num_generations), widths=0.6, patch_artist=True)
    
    # Apply colors
    colors = plt.cm.viridis(np.linspace(0, 1, num_generations))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    # Customize other box plot elements to match the colors
    for whisker, color in zip(box['whiskers'], np.repeat(colors, 2)):
        if isinstance(color, np.float64):
            color = '0'
        whisker.set_color(color)
    for cap, color in zip(box['caps'], np.repeat(colors, 2)):
        if isinstance(color, np.float64):
            color = '0'
        cap.set_color(color)
    for median, color in zip(box['medians'], colors):
        if isinstance(color, np.float64):
            color = '0'
        median.set_color('black')
    for flier, color in zip(box['fliers'], colors):
        if isinstance(color, np.float64):
            color = '0'
        flier.set(markeredgecolor=color)
    
    plt.xlabel('Generation')
    plt.ylabel('Average Distance Between Prompts')
    plt.title('Progression of Average Distance Over Generations')
    plt.grid(True)
    
    # Adjust x-ticks to show only integers
    plt.xticks(ticks=range(num_generations), labels=[int(gen) for gen in range(num_generations)])
    plt.savefig(f'Results/LKD/3_classes_distances.png')
    plt.show()
