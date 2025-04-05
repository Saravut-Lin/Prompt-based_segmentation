import pandas as pd
import matplotlib.pyplot as plt
import os

# Read the CSV files
training_loss_df = pd.read_csv('/Users/saravut_lin/EDINBURGH/Semester_2/ComV/Mini-Project/Prompt-based_segmentation/evaluate/csv_trainingLoss40.csv')
valid_loss_df = pd.read_csv('/Users/saravut_lin/EDINBURGH/Semester_2/ComV/Mini-Project/Prompt-based_segmentation/evaluate/csv_validLoss40.csv')

# Create a figure and axes
plt.figure(figsize=(10, 6))

# Plot the training loss
plt.plot(training_loss_df['Step'], training_loss_df['Value'], 
         label='Training Loss', color='blue', marker='o', markersize=4)

# Plot the validation loss
plt.plot(valid_loss_df['Step'], valid_loss_df['Value'], 
         label='Validation Loss', color='red', marker='x', markersize=4)

# Add labels and title
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Time')
plt.legend()

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Ensure the plot looks nice
plt.tight_layout()

# Save the figure
output_path = 'loss_curves.png'
plt.savefig(output_path, dpi=300)
print(f"Plot saved to {os.path.abspath(output_path)}")

# Show the plot
plt.show()