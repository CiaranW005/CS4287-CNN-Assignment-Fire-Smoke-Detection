"""Compare custom ResNet-50 feature dimensions with torchvision.

Run from the project root:
    .venv/bin/python scripts/test_resnet_dims.py
    .venv/bin/python scripts/test_resnet_dims.py --sizes 224x224 640x640 --batch-size 2

Checks shapes and values after copying torchvision weights into the custom model.
No pretrained weights are downloaded.
"""

import argparse
import importlib.util
from pathlib import Path

import torch
from torchvision.models import resnet50


def load_backbone():
    # Avoid fine_tune/__init__.py, which imports the unfinished HybridModel.
    path = Path(__file__).resolve().parents[1] / "src/fine_tune/models/resnet.py"
    spec = importlib.util.spec_from_file_location("custom_resnet", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load backbone from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ResNetBackbone()


def parse_size(value):
    try:
        height, width = map(int, value.lower().split("x"))
        if height > 0 and width > 0:
            return height, width
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("Use positive dimensions such as 640x640")


def copy_reference_weights(custom, reference):
    """Map custom parameter names to torchvision, including BatchNorm buffers."""
    mapped = {}
    reference_state = reference.state_dict()
    for name in custom.state_dict():
        if name.startswith("stem.conv."):
            reference_name = name.replace("stem.conv.", "conv1.", 1)
        elif name.startswith("stem.batch."):
            reference_name = name.replace("stem.batch.", "bn1.", 1)
        else:
            reference_name = name.replace("stage", "layer", 1)
            for index in (1, 2, 3):
                reference_name = reference_name.replace(f".batch{index}.", f".bn{index}.")
            reference_name = reference_name.replace(".shortcut.", ".downsample.")
        mapped[name] = reference_state[reference_name]
    custom.load_state_dict(mapped, strict=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", type=parse_size, nargs="+",
        default=[(224, 224), (640, 640), (225, 321)],
    )
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    torch.set_num_threads(2)
    torch.manual_seed(0)
    custom = load_backbone().eval()
    reference = resnet50(weights=None).eval()
    copy_reference_weights(custom, reference)
    custom_shapes = {}
    handles = []

    def capture(name):
        def hook(_module, _inputs, output):
            custom_shapes[name] = tuple(output.shape)
        return hook

    for name in ("stem", "stage1", "stage2", "stage3", "stage4"):
        handles.append(getattr(custom, name).register_forward_hook(capture(name)))

    try:
        with torch.inference_mode():
            for height, width in args.sizes:
                inputs = torch.randn(args.batch_size, 3, height, width)
                custom_shapes.clear()
                outputs = custom(inputs)
                features = reference.maxpool(reference.relu(reference.bn1(reference.conv1(inputs))))
                reference_shapes = {"stem": tuple(features.shape)}
                reference_outputs = []
                for index in range(1, 5):
                    features = getattr(reference, f"layer{index}")(features)
                    reference_shapes[f"stage{index}"] = tuple(features.shape)
                    if index >= 2:
                        reference_outputs.append(features)

                print(f"\nInput: {tuple(inputs.shape)}")
                for name, expected in reference_shapes.items():
                    actual = custom_shapes[name]
                    if actual != expected:
                        raise AssertionError(
                            f"{name}: custom={actual}, torchvision={expected}"
                        )
                    print(f"  {name:6s}: {actual}  PASS")

                if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
                    raise AssertionError("Backbone must return three features: P3, P4, P5")
                for name, output, expected in zip(("P3", "P4", "P5"), outputs, reference_outputs):
                    if output.shape != expected.shape:
                        raise AssertionError(
                            f"{name}: custom={tuple(output.shape)}, torchvision={tuple(expected.shape)}"
                        )
                    matches = torch.allclose(output, expected, atol=1e-5, rtol=1e-4)
                    max_error = (output - expected).abs().max().item()
                    print(f"  {name}: allclose={matches}, max_abs_error={max_error:.8g}")
                    torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-4)
                print("  P3/P4/P5 return shapes: PASS")
    finally:
        for handle in handles:
            handle.remove()

    print("\nAll backbone dimensions and values match torchvision ResNet-50.")


if __name__ == "__main__":
    main()
