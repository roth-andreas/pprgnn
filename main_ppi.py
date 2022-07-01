import json
from train import trainer_ppi

load_config = True
dataset = 'PPI'
for params in [
    {
        'dataset': dataset,
        'epochs': 10000,
        'self_loops': False,
        'type': 'pprgnn',
        'eps': 0.25,
        'lr': 0.005,
        'clip': 0.5,
        'weight_decay': 0,
        'dropout': 0.5,
        'width': 256,
        'schedule': True
    }
]:
    if load_config:
        with open(f'configs/{params["type"]}/{dataset}.json') as config_file:
            loaded_config = json.load(config_file)
        params = params | loaded_config

    metrics = trainer_ppi.train_model(**params)

    print(f'Results for {params["type"]}: {metrics}')
