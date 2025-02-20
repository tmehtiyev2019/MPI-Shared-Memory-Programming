import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read the consolidated CSV file
df = pd.read_csv('consolidated_results.csv')

# Matrix names (in the order they appear in the bash script)
matrices = ['D6-6', 'dictionary28', 'Ga3As3H12', 'bfly', 'pkustk14', 'roadNet-CA']

# Define versions with their configurations to match CSV format
versions = [
    ('MPI', 'N8_n8'),
    ('OpenMP', 'N1_n8'),
    ('Hybrid', 'N8_n8'),
    ('Hybrid', 'N4_n8'),
    ('Hybrid', 'N2_n8')
]

# Create labels for the legend with more readable format
version_labels = [
    'MPI (-N 8 -n 8)',
    'OpenMP (-N 1 -n 8)',
    'Hybrid (-N 8 -n 8)',
    'Hybrid (-N 4 -n 8)',
    'Hybrid (-N 2 -n 8)'
]

# Plotting setup
plt.figure(figsize=(15, 8))
x = np.arange(len(matrices))
width = 0.15

# Plot bars for each version
for i, (version, config) in enumerate(versions):
    data = []
    for matrix in matrices:
        time = df[(df['Version'] == version) & 
                 (df['Configuration'] == config) & 
                 (df['Matrix'] == matrix)]['Execution_Time_ms'].values[0]
        data.append(time)
    
    plt.bar(x + i*width, data, width, label=version_labels[i])

plt.xlabel('Matrices')
plt.ylabel('Runtime (ms)')
plt.title('SpMV Performance Comparison')
plt.xticks(x + width*2, matrices, rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save plot
plt.savefig('perf-cmp.jpg', dpi=300, bbox_inches='tight')
plt.close()