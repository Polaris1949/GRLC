import argparse
import os

DATASET_SETTINGS = {
    'cora': {
        'class': 'Planetoid',
        'name': 'Cora',
    },
    'citeseer': {
        'class': 'Planetoid',
        'name': 'CiteSeer',
    },
    'av2': {
        'prog': 'train_av2.py',
        'class': 'Yamai',
        'name': 'Argoverse2',
        'extra': '--data-root .',
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=DATASET_SETTINGS.keys())
    args = parser.parse_args()
    mode = args.mode
    setting = DATASET_SETTINGS[mode]
    prog = setting.get('prog', 'train.py')
    epochs = setting.get('epochs', 1000)
    class_ = setting['class']
    name = setting['name']
    ctkey = 'classification'
    extra = setting.get('extra', '')
    cmd = f'python {prog} --epochs {epochs} --dataset-class {class_} --dataset-name {name} --custom-key {ctkey} {extra}'
    print(f"Executing command: {cmd}")
    ec = os.system(cmd)
    print(f"Command exit code: {ec}")
