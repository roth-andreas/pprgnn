import json

from train import trainer_amazon

load_config = False
dataset = 'Amazon'
for portion in [0.05, 0.06, 0.07, 0.08, 0.09]:
    params = {
        'dataset': dataset,
        'epochs': 10000,
        'self_loops': False,
        'type': 'pprgnn',
        'eps': 0.06,
        'lr': 0.01,
        'clip': 0.5,
        'weight_decay': 1e-8,
        'dropout': 0.5,
        'width': 128,
        'schedule': True,
        'portion': portion
    }
    if load_config:
        with open(f'configs/{params["type"]}/{dataset}.json') as config_file:
            loaded_config = json.load(config_file)
        params = params | loaded_config

    metrics = trainer_amazon.train_model(**params)

    print(f'Results for {params["type"]}: {metrics}')
