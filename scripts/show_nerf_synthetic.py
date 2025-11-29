import json
import numpy as np

datasets = ['mic', 'chair', 'ship', 'materials', 'lego', 'drums', 'ficus', 'hotdog']
result_psnr = []
result_ssim = []
result_lpips = []

base_dir = "output/release_version_nerf_synthetic"


for dataset in datasets:
    result_dir = f'{base_dir}/{dataset}/results.json'
    with open(result_dir, 'r') as f:
        data = json.load(f)['ours_30000']
        result_psnr.append(data['PSNR'])
        result_ssim.append(data['SSIM'])
        result_lpips.append(data['LPIPS'])

# Compute means
mean_psnr = np.mean(result_psnr)
mean_ssim = np.mean(result_ssim)
mean_lpips = np.mean(result_lpips)

# Append means to each metric list
result_psnr.append(mean_psnr)
result_ssim.append(mean_ssim)
result_lpips.append(mean_lpips)
datasets_with_mean = datasets + ['mean']

# Format and save output
output_lines = []
output_lines.append("Dataset: " + ' | '.join(datasets_with_mean))
output_lines.append("PSNR:    " + ' | '.join([f'{v:.2f}' for v in result_psnr]))
output_lines.append("SSIM:    " + ' | '.join([f'{v:.4f}' for v in result_ssim]))
output_lines.append("LPIPS:   " + ' | '.join([f'{v:.4f}' for v in result_lpips]))

for line in output_lines:
    print(line)

# Save to file
with open(base_dir + '/results_summary.txt', 'w') as f:
    f.write('\n'.join(output_lines))


final_psnr_str = ' & '.join([f'{v:.2f}' for v in result_psnr] + [f'{mean_psnr:.2f}'])
final_ssim_str = ' & '.join([f'{v:.2f}' for v in result_ssim] + [f'{mean_ssim:.2f}'])
final_lpips_str = ' & '.join([f'{v:.2f}' for v in result_lpips] + [f'{mean_lpips:.2f}'])

print(f'PSNR:     {final_psnr_str} \\\\')
print(f'SSIM:     {final_ssim_str} \\\\')
print(f'LPIPS:    {final_lpips_str} \\\\')

with open(base_dir + '/results_summary_latex.txt', 'w') as f:
    f.write(f'PSNR:     {final_psnr_str} \\\\ \n')
    f.write(f'SSIM:     {final_ssim_str} \\\\ \n')
    f.write(f'LPIPS:    {final_lpips_str} \\\\ \n')