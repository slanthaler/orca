import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import os
import pprint

# pipelines define the execution logic
from orca.pipelines import build_pipeline


@hydra.main(version_base=None, config_path="config", config_name="test")
def train(config: DictConfig):
    config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    
    print('running with config:')
    pprint.pprint(config_dict)

    with wandb.init(
            project=config.wandb.project,
            mode=config.wandb.mode, 
            config=config_dict,
            group=config.pipeline.type 
        ):
        print(f"Running Experiment with pipeline: {config.pipeline.type}")
        print(f"Working directory: {os.getcwd()}") # Hydra changes cwd by default!
        pipeline = build_pipeline(config.pipeline, config.evaluation)
        metrics = pipeline.run()

        print("Training completed. Metrics:")
        pprint.pprint(metrics)
        wandb.log(metrics)


if __name__ == "__main__":
    train()