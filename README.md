# FedDAA: Dynamic Client Clustering for Concept Drift Adaptation in Federated Learning



## Environment

1. Create the conda environment (installs PyTorch 2.4 with CUDA 11.8 support):
   ```
   conda env create -f environment.yml
   ```
2. Activate it:
   ```
   conda activate feddaa
   ```
3. Install the remaining pure-Python packages:
   ```
   pip install -r requirements.txt
   ```

## Data

Please split data by the following scripts. 


- CIFAR-10
```
python create_c/make_cifar_c-60_client-simple2-iid-4concept-change-name-version2.py
```

- CIFAR-100
```
python create_c/make_cifar_100_c-60_client-simple2-iid-4concept-change-name-version2.py
```

- Fashion-MNIST
```
python create_c/make_fmnist_c-60_client-simple2-iid-4concept-change-name-version2.py
```

## Commands
- CIFAR-10
```
python FedDAA_CIFAR10.py cifar10-c fedrc_store_history --n_learners 2 --bz 128 --lr 0.06 --lr_scheduler constant --log_freq 1 --optimizer sgd --seed 1 --verbose 1 --T 6 --n_rounds 40 --device 0 --sampling_rate 0.5 --suffix T_6-client_60-FedDAA-CIFAR-10
```
- CIFAR-100
```
python FedDAA_CIFAR100.py cifar100-c fedrc_store_history --n_learners 2 --bz 128 --lr 0.06 --lr_scheduler constant --log_freq 1 --optimizer sgd --seed 1 --verbose 1 --T 6 --n_rounds 40 --device 0 --sampling_rate 0.5 --suffix T_6--client_60-FedDAA-CIFAR-100
```
- Fashion-MNIST
```
python FedDAA_Fashion_MNIST.py fmnist-c fedrc_store_history --n_learners 2 --bz 128 --lr 0.06 --lr_scheduler constant --log_freq 1 --optimizer sgd --seed 1 --verbose 1 --T 6 --n_rounds 40 --device 0 --sampling_rate 0.5 --suffix T_6-client_60-FedDAA-Fashion-MNIST


```

python FedDAA_CIFAR10.py cifar10-c fedrc_store_history --data_dir cifar10-c-60_client-multiclass-drift --drift_detector metrics --diagnosis_mode multiclass --adaptive_method median_mad --adaptive_warmup 1 --adaptive_window 3 --adaptive_k 2.5 --respect_drift_types --n_learners 2 --bz 128 --lr 0.06 --lr_scheduler constant --log_freq 5 --optimizer sgd --seed 1 --verbose 1 --T 3 --n_rounds 40 --device 0 --sampling_rate 0.5 --suffix multiclass_drift_T_6-client_60-FedDAA-CIFAR-10


