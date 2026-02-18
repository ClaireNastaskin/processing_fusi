import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# EDIT YOUR DATA HERE
# =============================================================================

# Condition labels (x-axis)
conditions = [
    'shallow 3MHz',
    'shallow 2MHz',
    'deep 3MHz',
    'very deep 3MHz'
]

# Values for each condition (must match order of conditions above)
pinky_values = [7.4, 5.9, 7.1, 7.2]
middle_values = [6.1, 5.4, 7.0, 7.7]
thumb_values = [9.1, 6.5, 6.0, 6.0]

# Output file path
output_file = '/Users/clairenastaskin/Documents/Anise/Experiments/max_zscore_plots.png'

# =============================================================================
# PLOT SETTINGS (optional to modify)
# =============================================================================
pinky_color = '#E74C3C'  # Red
middle_color = '#2ECC71'  # Gree    n
thumb_color = '#3498DB'  # Blue 
marker_size = 120
y_label = 'Max Z-Score'
title = 'Max Z-Score by Condition'

# =============================================================================
# PLOTTING CODE
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# X positions for dots
x = np.arange(len(conditions))
offset = 0.1  # Offset to separate pinky and thumb dots

# Plot pinky (dots)
ax.scatter(x - offset, pinky_values, s=marker_size, c=pinky_color,
           label='Pinky', edgecolors='black', linewidth=1, zorder=3)

# Plot middle (dots)
ax.scatter(x, middle_values, s=marker_size, c=middle_color,
           label='Middle', edgecolors='black', linewidth=1, zorder=3)

# Plot thumb (dots)
ax.scatter(x + offset, thumb_values, s=marker_size, c=thumb_color,
           label='Thumb', edgecolors='black', linewidth=1, zorder=3)

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(conditions, rotation=45, ha='right')
ax.set_ylabel(y_label, fontsize=12)
ax.set_title(title, fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(3.1, max(max(pinky_values), max(middle_values), max(thumb_values)) + 1)

plt.tight_layout()
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.show()

print(f"Plot saved to: {output_file}")
