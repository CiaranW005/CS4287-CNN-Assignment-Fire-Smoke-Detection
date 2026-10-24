"""Print and validate torchvision -> custom ResNet-50 weight names.

Run: .venv/bin/python tests/test_resnet_weight_names.py
Use --prefix stage1.0 to inspect one block, or --parameters-only to hide buffers.
No pretrained weights are downloaded. BatchNorm buffers are included by default
because running_mean, running_var and num_batches_tracked must also be mapped.
"""

import argparse

import torch
from torchvision.models import resnet50

from test_resnet_dims import load_backbone


def torchvision_name(custom_name):
    """Translate a custom state_dict key into its torchvision source key."""
    if custom_name.startswith("stem.conv."):
        return custom_name.replace("stem.conv.", "conv1.", 1)
    if custom_name.startswith("stem.batch."):
        return custom_name.replace("stem.batch.", "bn1.", 1)

    name = custom_name.replace("stage", "layer", 1)
    for index in (1, 2, 3):
        name = name.replace(f".batch{index}.", f".bn{index}.")
    return name.replace(".shortcut.", ".downsample.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="", help="Filter displayed custom names")
    parser.add_argument("--parameters-only", action="store_true")
    args = parser.parse_args()

    custom = load_backbone()
    reference = resnet50(weights=None)
    custom_state = custom.state_dict()
    reference_state = reference.state_dict()
    parameters = dict(custom.named_parameters())
    mapped = {}
    used = set()

    print(f"{'Torchvision source':<48} {'Custom destination':<48} {'Shape':<22} Kind")
    print("-" * 132)
    for custom_name, destination in custom_state.items():
        source_name = torchvision_name(custom_name)
        if source_name in used:
            raise AssertionError(f"Duplicate source mapping: {source_name}")
        if source_name not in reference_state:
            raise AssertionError(f"Missing source: {source_name} -> {custom_name}")
        source = reference_state[source_name]
        if source.shape != destination.shape or source.dtype != destination.dtype:
            raise AssertionError(
                f"Incompatible tensors: {source_name} {source.shape}/{source.dtype} -> "
                f"{custom_name} {destination.shape}/{destination.dtype}"
            )
        mapped[custom_name] = source
        used.add(source_name)
        kind = "parameter" if custom_name in parameters else "buffer"
        if custom_name.startswith(args.prefix) and (not args.parameters_only or kind == "parameter"):
            print(f"{source_name:<48} {custom_name:<48} {str(tuple(source.shape)):<22} {kind}")

    unused = set(reference_state) - used
    if unused != {"fc.weight", "fc.bias"}:
        raise AssertionError(f"Unexpected unmapped torchvision keys: {sorted(unused)}")

    custom.load_state_dict(mapped, strict=True)
    for name, actual in custom.state_dict().items():
        if not torch.equal(actual, reference_state[torchvision_name(name)]):
            raise AssertionError(f"Weight copy failed: {name}")

    print(f"\nPASS: mapped and copied all {len(mapped)} state_dict entries.")
    print("Excluded torchvision classifier: fc.weight, fc.bias (backbone has no classifier).")
    print("Display filters do not limit validation: the entire backbone is always checked.")


if __name__ == "__main__":
    main()
