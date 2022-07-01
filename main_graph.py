import json
from train import trainer_graph

load_config = True
for dataset in ['PTC', 'PROTEINS', 'MUTAG', 'COX2', 'NCI1']:
    params = {
        'dataset': dataset,
        'epochs': 10000,
        'self_loops': True,
        'type': 'pprgnn',
        'eps': 1.0,
        'lr': 0.01,
        'clip': 25,
        'weight_decay': 1e-6,
        'dropout': 0.5,
        'width': 64,
    }
    if load_config:
        with open(f'configs/{params["type"]}/{dataset}.json') as config_file:
            loaded_config = json.load(config_file)
        params = params | loaded_config

    metrics = trainer_graph.train_model(**params)
    print(f'10-fold results for {params["type"]}: {metrics}')
