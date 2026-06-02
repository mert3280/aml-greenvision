"""WS1 acceptance: the critical constants equal the guide's locked values."""

from greenvision import constants as C


def test_critical_constants():
    assert C.IMAGE_SIZE == 224
    assert C.NUM_CLASSES == 39  # D-17b: 38 leaf classes + Background_without_leaves.
    assert C.EFFICIENTNET_FEATURES == 1280
    assert C.DROPOUT_RATE == 0.2
    assert C.RESIZE_SIZE == 256


def test_imagenet_normalization_shape_and_values():
    assert len(C.IMAGENET_MEAN) == len(C.IMAGENET_STD) == 3
    assert C.IMAGENET_MEAN == (0.485, 0.456, 0.406)
    assert C.IMAGENET_STD == (0.229, 0.224, 0.225)


def test_seed_and_paths_defined():
    assert C.SEED == 42
    # Paths are derived from the repo root and end where we expect.
    assert C.CLASS_NAMES_PATH.name == "class_names.json"
    assert C.SPLITS_PATH.name == "splits.json"
    assert C.PHASE2_CKPT.name == "phase2_best.pt"
