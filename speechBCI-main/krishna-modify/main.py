# main.py
import os

import hydra
import wandb
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder

@hydra.main(config_path='configs', config_name='config')
def app(config):
    # set the visible device to the gpu specified in 'args' (otherwise tensorflow may claim all GPUs)
    if 'gpuNumber' in config:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        print(f'Setting CUDA_VISIBLE_DEVICES to {config["gpuNumber"]}')
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpuNumber'])

    if 'Slurm' in HydraConfig.get().launcher._target_:
        config.outputDir = './'
    print(f'Output dir {config.outputDir}')
    os.makedirs(config.outputDir, exist_ok=True)

    if 'wandb' in config and config.wandb.enabled:
        run = wandb.init(**config.wandb.setup,
                         config=OmegaConf.to_container(config, resolve=True),
                         sync_tensorboard=True,
                         resume=True)

    print(f"Initializing model type: {config.model.get('type', 'gru')}")
    nsd = NeuralSequenceDecoder(args=config)

    if config['mode'] == 'train':
        cer = nsd.train()
        return cer
    elif config['mode'] in ['inference', 'infer']:
        nsd.inference()
    else:
        raise ValueError(f"Unknown mode: {config['mode']}")

if __name__ == "__main__":
    app()
