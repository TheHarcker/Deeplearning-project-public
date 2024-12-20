#!/bin/bash
# embedded options to bsub - start with #BSUB
# -- our name ---
#BSUB -J DLProjectWithAllOptimizations
# -- choose queue --
#BSUB -q gpuv100
# -- specify that we need 4GB of memory per core/slot --
# so when asking for 4 cores, we are really asking for 4*4GB=16GB of memory 
# for this job. 
#BSUB -R "rusage[mem=3GB]"
# -- Notify me by email when execution begins --
#BSUB -B
# -- Notify me by email when execution ends   --
#BSUB -N
# -- Output File --
#BSUB -o Output_%J.out
# -- Error File --
#BSUB -e Output_%J.err
# -- estimated wall clock time (execution time): hh:mm -- 
#BSUB -W 18:00 
# -- Number of cores requested -- 
#BSUB -n 4
# -- Specify the distribution of the cores: on a single node --
#BSUB -R "span[hosts=1]"
# -- Test to see if we can be run faster by not setting exclusive_process
#
# -- end of LSF options -- 

module load python3 
module load cuda/12.5

echo "Start dependency installation"
python3 -m pip install torch tqdm numpy lightning torch-geometric torchvision rdkit scipy tabulate 
echo "Installed dependencies. Running script" 

mkdir -pv Results/"n="$layers"layers/job"$LSB_JOBID 
python3 -OO training.py --num_workers 3 --compile=True --num_epochs 1000 --num_message_passing_layers=$layers --job_id="n="$layers"layers/job"$LSB_JOBID > joboutput_$LSB_JOBID.out 2>&1 
