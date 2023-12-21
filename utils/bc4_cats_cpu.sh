#!/bin/env bash
#SBATCH --account=sscm012844
#SBATCH --job-name=cats_paper
#SBATCH --output=cats_paper
#SBATCH --error=cats_paper
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --time=1-00:00:00
#SBATCH --mem=100000M
#SBATCH --array=1-4

# Load the modules/environment
module purge
module load languages/anaconda3/3.7
conda init
source ~/.bashrc


# Define working directory
export WORK_DIR=/user/work/fo18103/WodCat

# Change into working directory
cd ${WORK_DIR}
source /user/work/fo18103/WodCat/venv/bin/activate

# Do some stuff
echo JOB ID: ${SLURM_JOBID}
echo PBS ARRAY ID: ${SLURM_ARRAY_TASK_ID}
echo Working Directory: $(pwd)

cmds=('ml.py --study-id cat --output-dir /user/work/fo18103/Cats/data_test/10000_10_060_001/rbf/QN_LeaveOneOut --dataset-filepath /user/work/fo18103/Cats/data_test/10000_10_060_001/dataset/samples.csv --n-job 6 --cv LeaveOneOut --preprocessing-steps QN --class-healthy-label 0.0 --class-unhealthy-label 1.0 --meta-columns peak0_datetime --meta-columns label --meta-columns id --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --meta-columns max_sample --meta-columns n_peak --meta-columns w_size --meta-columns n_top --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --classifiers rbf ' 'ml.py --study-id cat --output-dir /user/work/fo18103/Cats/data_test/10000_10_060_002/rbf/QN_LeaveOneOut --dataset-filepath /user/work/fo18103/Cats/data_test/10000_10_060_002/dataset/samples.csv --n-job 6 --cv LeaveOneOut --preprocessing-steps QN --class-healthy-label 0.0 --class-unhealthy-label 1.0 --meta-columns peak0_datetime --meta-columns peak1_datetime --meta-columns label --meta-columns id --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --meta-columns max_sample --meta-columns n_peak --meta-columns w_size --meta-columns n_top --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --classifiers rbf ' 'ml.py --study-id cat --output-dir /user/work/fo18103/Cats/data_test/10000_10_120_001/rbf/QN_LeaveOneOut --dataset-filepath /user/work/fo18103/Cats/data_test/10000_10_120_001/dataset/samples.csv --n-job 6 --cv LeaveOneOut --preprocessing-steps QN --class-healthy-label 0.0 --class-unhealthy-label 1.0 --meta-columns peak0_datetime --meta-columns label --meta-columns id --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --meta-columns max_sample --meta-columns n_peak --meta-columns w_size --meta-columns n_top --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --classifiers rbf ' 'ml.py --study-id cat --output-dir /user/work/fo18103/Cats/data_test/10000_10_120_002/rbf/QN_LeaveOneOut --dataset-filepath /user/work/fo18103/Cats/data_test/10000_10_120_002/dataset/samples.csv --n-job 6 --cv LeaveOneOut --preprocessing-steps QN --class-healthy-label 0.0 --class-unhealthy-label 1.0 --meta-columns peak0_datetime --meta-columns peak1_datetime --meta-columns label --meta-columns id --meta-columns date --meta-columns health --meta-columns target --meta-columns age --meta-columns name --meta-columns mobility_score --meta-columns max_sample --meta-columns n_peak --meta-columns w_size --meta-columns n_top --individual-to-ignore MrDudley --individual-to-ignore Oliver_F --individual-to-ignore Lucy --classifiers rbf ')
# Execute code
echo ${cmds[${SLURM_ARRAY_TASK_ID}]}
python ${cmds[${SLURM_ARRAY_TASK_ID}]} > /user/work/fo18103/logs/cats_thesis_${SLURM_ARRAY_TASK_ID}.log
