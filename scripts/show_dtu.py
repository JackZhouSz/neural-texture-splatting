import json
import numpy as np
import os

scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
result_overall = []


base_dir = "output/release_version_dtu"



# --------------------------------------------------------------------------------

print(f"Processing: {os.path.basename(base_dir)}")

for scene in scenes:
    # Construct path based on your DTU structure
    json_file = f"{base_dir}/scan{scene}/test/ours_30000/tsdf/results.json"
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            # Extract the 'overall' metric
            result_overall.append(data["overall"])
    except Exception:
        # Handle missing files or errors by appending NaN
        result_overall.append(np.nan)

# Compute mean (ignoring NaNs)
mean_overall = np.nanmean(result_overall)

# Append mean to the result list for unified formatting
result_overall.append(mean_overall)
scene_labels = [str(s) for s in scenes] + ['Mean']

# --- Format Outputs ---

def fmt(val):
    """Helper to format numbers or handle missing data"""
    if np.isnan(val):
        return "---"
    return f"{val:.3f}"

# 1. Text / Console Output
output_lines = []
output_lines.append("Scans:   " + ' | '.join(scene_labels))
output_lines.append("Overall: " + ' | '.join([fmt(v) for v in result_overall]))

print("-" * 20)
for line in output_lines:
    print(line)
print("-" * 20)

# Save Text Summary
save_txt_path = os.path.join(base_dir, 'results_summary.txt')
with open(save_txt_path, 'w') as f:
    f.write('\n'.join(output_lines))
    print(f"Saved text summary to: {save_txt_path}")

# 2. LaTeX Output
final_latex_str = ' & '.join([fmt(v) for v in result_overall])
latex_line = f'Overall:  {final_latex_str} \\\\'

print("\n--- LaTeX String ---")
print(latex_line)

# Save LaTeX Summary
save_latex_path = os.path.join(base_dir, 'results_summary_latex.txt')
with open(save_latex_path, 'w') as f:
    f.write(latex_line + '\n')
    print(f"Saved LaTeX summary to: {save_latex_path}")