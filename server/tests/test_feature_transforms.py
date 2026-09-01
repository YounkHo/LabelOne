from __future__ import annotations

import numpy as np
import pytest

from labelone.errors import ModelRuntimeError
from labelone.models.features import FeatureTransformOptions, feature_preview_image, transform_feature


def test_mean_minmax_gain_gamma_and_spatial_scale_are_applied() -> None:
    tensor = np.array(
        [[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]],
        dtype=np.float32,
    )
    options = FeatureTransformOptions(
        projection="mean",
        normalization="minmax",
        spatial_scale=2,
        interpolation="nearest",
        gain=2,
        gamma=1,
    )

    result = transform_feature(tensor, options)

    assert result.shape == (1, 1, 4, 4)
    assert float(result.min()) == 0
    assert float(result.max()) == pytest.approx(2)
    assert result[0, 0, 0, 0] == result[0, 0, 1, 1]


def test_token_grid_and_pca_projection_are_deterministic() -> None:
    tensor = np.arange(1 * 4 * 3, dtype=np.float32).reshape(1, 4, 3)
    grid = transform_feature(tensor, FeatureTransformOptions(projection="token_grid"))
    first = transform_feature(tensor, FeatureTransformOptions(projection="pca1"))
    second = transform_feature(tensor, FeatureTransformOptions(projection="pca1"))

    assert grid.shape == (1, 1, 2, 2)
    assert first.shape == (1, 1, 2, 2)
    np.testing.assert_allclose(first, second)


def test_scaled_feature_budget_is_enforced() -> None:
    tensor = np.zeros((1, 4, 20, 20), dtype=np.float32)
    with pytest.raises(ModelRuntimeError, match="output budget"):
        transform_feature(
            tensor,
            FeatureTransformOptions(spatial_scale=4, max_output_elements=1_000),
        )


def test_pca_minmax_bicubic_transform_cleans_non_finite_extreme_values() -> None:
    maximum = np.finfo(np.float32).max
    tensor = np.linspace(-1, 1, 1 * 32 * 8 * 8, dtype=np.float32).reshape(1, 32, 8, 8)
    tensor[0, 0, 0, 0] = maximum
    tensor[0, 1, 0, 0] = -maximum
    tensor[0, 2, 0, 0] = np.nan
    tensor[0, 3, 0, 0] = np.inf

    result = transform_feature(
        tensor,
        FeatureTransformOptions(
            projection="pca1",
            normalization="minmax",
            interpolation="bicubic",
            spatial_scale=4,
            gain=1.8,
            gamma=0.75,
            clip_percentiles=(1, 99),
        ),
    )

    assert result.shape == (1, 1, 32, 32)
    assert np.all(np.isfinite(result))


def test_vector_and_matrix_features_have_bounded_real_previews() -> None:
    vector = transform_feature(
        np.linspace(-2, 2, 1_000, dtype=np.float32).reshape(1, 1_000),
        FeatureTransformOptions(projection="pca1", normalization="minmax", spatial_scale=8),
    )
    matrix = transform_feature(
        np.arange(12 * 4, dtype=np.float32).reshape(12, 4),
        FeatureTransformOptions(projection="mean", normalization="minmax"),
    )

    vector_preview = feature_preview_image(vector)
    matrix_preview = feature_preview_image(matrix)

    assert vector.shape == (1_000,)
    assert vector_preview is not None and vector_preview.size == (512, 128)
    assert matrix.shape == (12,)
    assert matrix_preview is not None and matrix_preview.size == (512, 128)
