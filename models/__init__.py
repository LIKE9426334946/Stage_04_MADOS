from .example_model import ExampleModel


# Add another model here after implementing its nn.Module in models/.
MODEL_REGISTRY = {"example": ExampleModel}


def build_model(model_config, in_channels, num_classes):
    name = model_config["name"]
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Available models: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](
        in_channels=in_channels, num_classes=num_classes,
        **model_config.get("kwargs", {}),
    )
