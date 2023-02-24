#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Test_Grayscale_1
#SBATCH --time=01:00:00                         #Request hr:min:sec
#SBATCH --ntasks=1                              #Request 1 task
#SBATCH --mem=5G                               #Request 2560MB (2.5GB) per node
#SBATCH --output=UU.%j                          #Send stdout/err to "Example5Out.[jobID]"
#SBATCH --gres=gpu:1	                        #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu                         #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=zacharybottorff              #Set billing account
##SBATCH --mail-type=ALL                        #Send email on all job events
##SBATCH --mail-user=zacharybottorff@tamu.edu   #Send all emails to email_address 

#First Executable Line
source tdenv/bin/activate
module restore dl
module load GCC/10.2.0  CUDA/11.1.1  OpenMPI/4.0.5
module load torchvision/0.10.0-PyTorch-1.9.0
module load matplotlib/3.3.3
python little_exm_pe.py 

