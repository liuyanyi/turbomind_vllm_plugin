

def register():
    from vllm.model_executor.layers.linear import WEIGHT_LOADER_V2_SUPPORTED

    # Register the quantization method
    from .turbomind_awq import (  # noqa: F401
        AWQTurbomindConfig,
        AWQTurbomindLinearMethod,
    )

    # This is needed for correct weight loading
    WEIGHT_LOADER_V2_SUPPORTED.append("AWQTurbomindLinearMethod")
