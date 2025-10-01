# Plots before

# Hardcoding from actual 16X16, cause of labeling this is before now it works.. 5x5 in validaiton itself

# src/generate_final_plots.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
# Define the 5 standard AAMI classes we will map to
AAMI_CLASSES = ['N', 'S', 'V', 'F', 'Q']



FINAL_CM_NOISY = np.array([
    [1595,    2,    1,   0,   37], # True N
    [  83,   28,    3,   0,   14], # True S
    [  37,    8,  135,   0,   18], # True V
    [   2,    0,    0,   0,    0], # True F
    [   0,    0,    0,   0,    0]  # True Q
], dtype=int)

FINAL_CM_L1_DENOISED = np.array([
    [1632,    3,    0,   0,    0], # True N
    [  54,   67,    6,   0,    1], # True S
    [   2,    2,  194,   0,    0], # True V
    [   2,    0,    0,   0,    0], # True F
    [   0,    0,    0,   0,    0]  # True Q
], dtype=int)


FINAL_CM_L1_GRAD_DENOISED = np.array([
    [1634,    1,    0,   0,    0], # True N
    [  45,   78,    4,   0,    1], # True S
    [   1,    1,  196,   0,    0], # True V
    [   2,    0,    0,   0,    0], # True F
    [   0,    0,    0,   0,    0]  # True Q
], dtype=int)


FINAL_CM_STPC_DENOISED = np.array([
    [1632,    3,    0,   0,    0], # True N
    [  54,   67,    6,   0,    1], # True S
    [   2,    2,  194,   0,    0], # True V
    [   2,    0,    0,   0,    0], # True F
    [   0,    0,    0,   0,    0]  # True Q
], dtype=int)

FINAL_CM_CLEAN = np.array([
    [1633,    2,    0,   0,    0], # True N
    [  41,   87,    0,   0,    0], # True S
    [   1,    0,  197,   0,    0], # True V
    [   2,    0,    0,   0,    0], # True F
    [   0,    0,    0,   0,    0]  # True Q
], dtype=int)


def plot_and_save_cm(cm_data, title, filename):
    """Generates a clean heatmap plot for a confusion matrix and saves it."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_data,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=AAMI_CLASSES,
        yticklabels=AAMI_CLASSES,
        annot_kws={"size": 14} # Make annotations larger
    )
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved simplified plot to: {filepath}")


def main():
    """Main function to generate final plots from all experimental conditions."""
    print("--- Generating Final, Simplified 5x5 AAMI Confusion Matrices ---")

    # Dictionary of all conditions to plot
    matrices_to_plot = {
        "Classifier on Noisy Signal": (FINAL_CM_NOISY, "final_cm_noisy.png"),
        "Classifier on Denoised Signal (L1 Only)": (FINAL_CM_L1_DENOISED, "final_cm_l1_only_denoised.png"),
        "Classifier on Denoised Signal (L1+Grad)": (FINAL_CM_L1_GRAD_DENOISED, "final_cm_l1_grad_denoised.png"),
        "Classifier on Denoised Signal (Full STPC)": (FINAL_CM_STPC_DENOISED, "final_cm_stpc_full_denoised.png"),
        "Classifier on Clean Signal (Upper Bound)": (FINAL_CM_CLEAN, "final_cm_clean.png"),
    }

    for title, (matrix, filename) in matrices_to_plot.items():
        plot_and_save_cm(matrix, title, filename)
        
    print("\n--- All plots generated successfully! ---")
    print("You now have a complete set of simplified CMs for your ablation study.")


if __name__ == "__main__":
    main()