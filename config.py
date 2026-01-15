"""Configuration & Hyperparameters"""

# === ENVIRONMENT ===
ENV_CONFIG = {
    'n_firms': 3,
    'n_households': 30,
    'max_steps': 100
}

# === TRAINING ===
TRAINING_CONFIG = {
    'learning_rate': 0.0003,
    'train_batch_size': 4000,
    'sgd_minibatch_size': 128,
    'num_sgd_iter': 10,
    'entropy_coeff': 0.01,
    'num_workers': 4,
    'total_iterations': 100
}

# === AGENT PARAMETERS ===
FIRM_CONFIG = {
    'start_kapital_min': 10000,
    'start_kapital_max': 20000,
    'start_preis': 100,
    'start_lohn': 50,
    'start_mitarbeiter': 10,
    'produktivitaet_min': 0.8,
    'produktivitaet_max': 1.2
}

HOUSEHOLD_CONFIG = {
    'start_vermoegen_min': 2000,
    'start_vermoegen_max': 8000,
    'sparquote_min': 0.1,
    'sparquote_max': 0.3,
    'konsumneigung_min': 0.7,
    'konsumneigung_max': 0.95
}
