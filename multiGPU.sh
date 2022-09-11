#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Test_Sml_Ray_SR15p
#SBATCH --time=5:00:00
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --mem=10G                  #Request 2560MB (2.5GB) per node
#SBATCH --output=UU.%j      #Send stdout/err to "Example5Out.[jobID]"
#SBATCH --gres=gpu:1	               #Request 2 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=l2194121700             #Set billing account to 123456
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=l2194121700@tamu.edu    #Send all emails to email_address 

#First Executable Line
module load GCC/10.2.0  CUDA/11.1.1  OpenMPI/4.0.5
module load torchvision/0.10.0-PyTorch-1.9.0
module load matplotlib/3.3.3
python little_exm.py 

