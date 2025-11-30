from orca.pipelines.torch_pipeline import TorchPipeline
from orca.pipelines.kernel_pipeline import KernelPipeline

_PIPELINE_REGISTRY = {
    "torch": TorchPipeline,
    "kernel": KernelPipeline,
}

def build_pipeline(pipeline_cfg, eval_cfg):
    '''
    Factory method to build and return the appropriate pipeline instance
    based on the provided configuration.
    '''
    pipeline_type = pipeline_cfg.type
    if pipeline_type not in _PIPELINE_REGISTRY:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")
    pipeline_class = _PIPELINE_REGISTRY[pipeline_type]
    print(pipeline_class)
    return pipeline_class(pipeline_cfg, eval_cfg)