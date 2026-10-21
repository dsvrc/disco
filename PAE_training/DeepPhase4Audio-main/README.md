# Learning Phase Manifolds in Music
The project aims to adapt the DeepPhase architecture for motion
(Stark et al.) [1] to learn the phase manifolds of musical pieces
in an unsupervised manner. We aim to invoke latent channels
that capture the non-linear periodicity in a song, from which we
can reconstruct the input. 


## Setup
### Prerequisites
Required Packages (see ```requirements.txt```):
- Python 3.12.3
- matplotlib==3.8.4
- numpy==2.1.3
- Plotting==0.0.7
- pytorch_lightning==2.4.0
- scikit_learn==1.4.2
- torch==2.5.1
- rich >= 10.2.2
  
Required modules on Slurm:
  - `gcc/11`
  - `cuda/12.1`

### Steps
- **Local development**: To use the repository locally, simply clone it and install the dependencies inside a virtual environment. 
- **Running on a Slurm cluster**: This code is also designed to be compatible with a Slurm cluster environment. Clone the repository in your cluster and run ```./setup.sh``` for automatically creating the environment and adding the modules.
- **How to run the code**: Run ./train_slurm.sh. Replace the config file inside train_slurm.sh with the config file for the model you want to run.


```python
bash train.sh

python -m src.scripts.train --config_file configs/paeflat200.yaml #reproduce the result from original test checkpoint

cd PAE_training/DeepPhase4Audio-main && \
python -m src.scripts.train --config_file configs/paedeep.yaml

cd PAE_training/DeepPhase4Audio-main && \
python -m src.scripts.train --config_file configs/pae_wave.yaml

cd PAE_training/DeepPhase4Audio-main && \
python -m src.scripts.train --config_file configs/vqpaethreetone_50eps_64emb.yaml

cd PAE_training/DeepPhase4Audio-main && \
python -m src.scripts.train --config_file configs/vqpaeTwinkle_50eps_64emb.yaml

cd PAE_training/DeepPhase4Audio-main && \
conda activate music2dance && \
python test_vqpae_full_audio.py

# This is the training setup used for the LDF model
cd PAE_training/DeepPhase4Audio-main && \
CUDA_VISIBLE_DEVICES=7 python -m src.scripts.train --config_file configs/vqpae_aistpp_1000eps_8emb.yaml

# Retrain on GTZAN dataset with the same settings
cd PAE_training/DeepPhase4Audio-main && \
CUDA_VISIBLE_DEVICES=7 python -m src.scripts.train --config_file configs/vqpae_gtzan_1000eps_8emb.yaml
```

