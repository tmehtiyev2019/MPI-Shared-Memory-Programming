import os
import pandas as pd
import matplotlib
# Force non-interactive backend
matplotlib.use('Agg')
# Disable underscore interpretation in text
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
import numpy as np
import colorsys

# Read the CSV data
df = pd.read_csv('final_results.csv')
print(f"Successfully loaded results with {len(df)} entries")

# Ensure Execution_Time_ms is numeric
df['Execution_Time_ms'] = pd.to_numeric(df['Execution_Time_ms'], errors='coerce')

# Matrix names (in the order they appear in the bash script)
matrices = ['D6-6', 'dictionary28', 'Ga3As3H12', 'bfly', 'pkustk14', 'roadNet-CA']

# Extract all unique version-configuration combinations from the data
version_configs = df.drop_duplicates(['Version', 'Configuration'])[['Version', 'Configuration']]
all_versions = list(zip(version_configs['Version'], version_configs['Configuration']))

# Create a list of MPI+CUDA versions
mpi_cuda_versions = [(v, c) for v, c in all_versions if 'MPI+CUDA' in v]

# Sort MPI+CUDA versions by the number of processes
def get_np_value(version_str):
    if 'MPI+CUDA' in version_str:
        return int(version_str.split('_np')[1].split('-')[0])
    return 0

mpi_cuda_versions.sort(key=lambda x: get_np_value(x[0]))

# Create final ordered version list
ordered_versions = [
    ('Sequential', 'N/A'),
    ('MPI', 'N8_n8'),
    ('OpenMP', 'N1_n8'),
    ('Hybrid', 'N8_n8'),
    ('Hybrid', 'N4_n8'),
    ('Hybrid', 'N2_n8'),
    ('CUDA', 'GPU')
] + mpi_cuda_versions

# Create readable labels
custom_labels = []
for version, config in ordered_versions:
    if version == 'Sequential':
        custom_labels.append('Sequential')
    elif version == 'MPI':
        custom_labels.append(f'MPI ({config})')
    elif version == 'OpenMP':
        custom_labels.append(f'OpenMP ({config})')
    elif version == 'Hybrid':
        custom_labels.append(f'MPI+OpenMP ({config})')
    elif version == 'CUDA':
        custom_labels.append('CUDA')
    elif 'MPI+CUDA' in version:
        np_value = version.split('_np')[1].split('-')[0]
        gpu_value = version.split('-G')[1]
        custom_labels.append(f'MPI+CUDA (np={np_value}, G={gpu_value})')

# Generate distinct colors for each implementation
def generate_distinct_colors(n):
    colors = []
    # Start with some predefined distinct colors for the main categories
    base_colors = [
        '#1f77b4',  # blue - Sequential
        '#2ca02c',  # green - MPI
        '#d62728',  # red - OpenMP
        '#9467bd',  # purple - Hybrid (N8_n8)
        '#8c564b',  # brown - Hybrid (N4_n8)
        '#e377c2',  # pink - Hybrid (N2_n8)
        '#ff7f0e',  # orange - CUDA
    ]
    
    colors.extend(base_colors[:min(len(base_colors), n)])
    
    # If we need more colors, generate them with good separation
    if n > len(base_colors):
        remaining = n - len(base_colors)
        for i in range(remaining):
            hue = i / remaining
            saturation = 0.7 + (i % 3) * 0.1  # Vary saturation slightly
            value = 0.9 - (i % 2) * 0.2       # Vary brightness slightly
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            hex_color = '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
            colors.append(hex_color)
    
    return colors

colors = generate_distinct_colors(len(ordered_versions))

# Plotting setup
plt.figure(figsize=(16, 10))
x = np.arange(len(matrices))
width = 0.8 / len(ordered_versions)

# Plot bars and store handles for custom legend
handles = []
for i, ((version, config), label, color) in enumerate(zip(ordered_versions, custom_labels, colors)):
    data = []
    for matrix in matrices:
        # Filter based on version and configuration
        filtered = df[(df['Version'] == version) & 
                    (df['Matrix'] == matrix) &
                    (df['Configuration'] == config)]
        
        if not filtered.empty:
            time = filtered['Execution_Time_ms'].values[0]
            data.append(float(time) if pd.notna(time) else np.nan)
        else:
            data.append(np.nan)
    
    # Plot only if we have valid data
    if not all(np.isnan(data)):
        bar = plt.bar(x + i*width - 0.4, data, width, color=color)
        handles.append(bar[0])
    else:
        # Create an empty/invisible bar for the legend
        # Important: When we create an empty bar, it returns an empty container
        # So we need to handle it differently
        invisible_bar = plt.Rectangle((0,0), 1, 1, fc=color)
        handles.append(invisible_bar)

plt.xlabel('Matrices', fontsize=14)
plt.ylabel('Runtime (ms)', fontsize=14)
plt.title('SpMV Performance Comparison Across All Implementations', fontsize=16)
plt.xticks(x, matrices, rotation=45, fontsize=12)

# Add legend with manually created labels
plt.legend(handles, custom_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

plt.grid(True, alpha=0.3)
plt.yscale('log')  # Use log scale for better visibility of differences
plt.tight_layout()

# Save plot
plt.savefig('perf-cmp-all-upt.jpg', dpi=300, bbox_inches='tight')
plt.close()

print("Performance comparison plot saved as perf-cmp-all.jpg")