import numpy as np

from anima_def_ghostfwl.inference.postprocess import (
    ghost_mask_from_labels,
    labels_to_point_cloud,
    threshold_predictions,
    write_point_cloud_artifact,
)


def test_threshold_predictions_uses_default_point_five() -> None:
    probabilities = np.zeros((2, 2, 2, 4), dtype=np.float32)
    probabilities[..., 3] = 0.49
    probabilities[..., 1] = 0.48
    labels = threshold_predictions(probabilities)
    assert np.all(labels == 0)


def test_postprocess_writes_point_cloud_artifact(tmp_path) -> None:
    labels = np.array(
        [
            [[0, 3], [1, 2]],
            [[3, 3], [1, 0]],
        ],
        dtype=np.int32,
    )
    ghost_mask = ghost_mask_from_labels(labels)
    points = labels_to_point_cloud(labels)
    artifact = write_point_cloud_artifact(tmp_path / "denoised_points.npy", points)

    assert ghost_mask.sum() == 3
    assert artifact.exists()
    assert np.load(artifact).shape[1] == 3
