import math

import numpy as np
import pandas as pd
import pytest

import attention_plasticity.head_analysis as head_analysis
from attention_plasticity import (
    analyze_head,
    compute_attention_plasticity,
    compute_first_component_residual_stats,
    compute_noise_normality_stats,
    compute_scalar_position_predictability,
    fit_multioutput_linear_regressor,
    make_pos_rotation,
    orientation_from_keys,
    stack_oriented_examples,
)


class DummySplit:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def to_pandas(self):
        # Return a copy so tests can't mutate shared state.
        return self._df.copy()


class DummyDataset:
    def __init__(self, df: pd.DataFrame):
        self._split = DummySplit(df)

    def __getitem__(self, key: str):
        if key != "train":
            raise KeyError(key)
        return self._split

    def to_pandas(self):
        return self._split.to_pandas()


def make_df(vectors, buckets, positions, sliding_window=None, example_ids=None):
    n = len(buckets)
    if sliding_window is None:
        sliding_window = [0] * n
    if example_ids is None:
        example_ids = np.arange(n)
    return pd.DataFrame(
        {
            "bucket": buckets,
            "example_id": example_ids,
            "position": positions,
            "vector": [np.asarray(v, dtype=np.float64) for v in vectors],
            "sliding_window": sliding_window,
        }
    )


def test_fit_multioutput_linear_regressor_handles_constant_positions():
    t = np.ones(4, dtype=np.float64)
    X = np.stack(
        [
            np.linspace(0.0, 1.5, num=4),
            np.linspace(-2.0, 2.0, num=4),
        ],
        axis=1,
    )

    alpha, beta, r2_components, r2_overall = fit_multioutput_linear_regressor(t, X)

    assert np.allclose(beta, 0.0)
    assert np.allclose(alpha, X.mean(axis=0))
    assert np.allclose(r2_components, 0.0)
    assert r2_overall == 0.0


def test_orientation_from_keys_sets_positive_sign_on_zero_means():
    vectors = [
        np.array([1.0, -2.0, 0.0]),
        np.array([2.0, -1.0, 0.0]),
        np.array([-3.0, 4.0, 0.0]),
    ]
    buckets = np.array([0, 1, 1])
    positions = np.array([0.0, 1.0, 2.0])
    df = make_df(vectors, buckets, positions)
    orient = orientation_from_keys(DummyDataset(df))

    assert np.all(orient[:2] == np.sign([1.5, -1.5]))
    assert orient[2] == 1.0  # zero-mean dimension defaults to +1


def test_make_pos_rotation_aligns_beta_with_axis():
    beta = np.array([0.0, 3.0, 4.0], dtype=np.float64)

    H = make_pos_rotation(beta)
    e1 = np.zeros(3)
    e1[0] = 1.0

    expected_dir = beta / np.linalg.norm(beta)
    assert np.allclose(H @ e1, expected_dir)
    assert np.allclose(H.T @ H, np.eye(3), atol=1e-12)


def test_make_pos_rotation_returns_identity_for_zero_beta():
    beta = np.zeros(4)
    H = make_pos_rotation(beta)
    assert np.allclose(H, np.eye(4))


def test_stack_oriented_examples_subsample_deterministic():
    vectors = [np.array([i, -i], dtype=np.float64) for i in range(6)]
    buckets = np.arange(6) % 2
    positions = np.arange(6, dtype=np.float64)
    df = make_df(vectors, buckets, positions)
    ds = DummyDataset(df)
    orient = np.array([1.0, -1.0])

    p1, b1, X1 = stack_oriented_examples(ds, orient, max_tokens=3, seed=42)
    p2, b2, X2 = stack_oriented_examples(ds, orient, max_tokens=3, seed=42)

    assert np.array_equal(p1, p2)
    assert np.array_equal(b1, b2)
    assert np.array_equal(X1, X2)
    assert np.allclose(X1[:, 0], X1[:, 1])  # Orientation flips the sign of dim 1.


def test_compute_scalar_position_predictability_improves_baseline():
    p = np.linspace(1.0, 5.0, num=20)
    X = np.stack([p, 0.5 * p], axis=1)
    beta = np.array([1.0, 0.5])

    r2_scalar, mae_baseline, mae_proj = compute_scalar_position_predictability(p, X, beta)

    assert r2_scalar > 0.95
    assert mae_proj < mae_baseline


def test_compute_first_component_residual_stats_handles_constant_position():
    p = np.ones(5, dtype=np.float64)
    X_rot = np.stack([np.linspace(-1.0, 1.0, num=5), np.zeros(5)], axis=1)
    sk, kt, pval = compute_first_component_residual_stats(p, X_rot)
    assert sk == pytest.approx(0.0, abs=1e-12)
    assert kt == pytest.approx(-1.2, abs=1e-12)
    assert pval == pytest.approx(1.0, abs=5e-4)


def test_compute_noise_normality_stats_low_samples_returns_defaults():
    X_rot = np.ones((2, 5), dtype=np.float64)
    stats = compute_noise_normality_stats(X_rot, max_dims=3, alpha=0.05, seed=0)
    assert stats == (0.0, 0.0, 1.0, 0.0)


def test_compute_attention_plasticity_expected_value():
    p_q = np.array([0.0, 1.0, 1.0])
    b_q = np.array([0, 1, 1])
    X_q_rot = np.array([[0.0], [1.0], [1.0]])
    alpha_rot = np.array([0.0])
    beta_rot = np.array([1.0])
    resid_var = np.array([1.0])

    p_k = np.array([0.0, 0.0])
    b_k = np.array([0, 0])
    X_k_rot = np.array([[1.0], [-1.0]])
    example_ids = np.array([0, 0])

    ap_overall, ap_pairs = compute_attention_plasticity(
        p_q=p_q,
        b_q=b_q,
        X_q_rot=X_q_rot,
        alpha_rot=alpha_rot,
        beta_rot=beta_rot,
        resid_var=resid_var,
        p_k=p_k,
        b_k=b_k,
        example_ids_k=example_ids,
        X_k_rot=X_k_rot,
        num_pairs_per_bucket=1,
        seed=0,
    )

    p_z = 0.5 * (1 + math.erf(1 / math.sqrt(2)))
    expected_pp = 4 * p_z * (1 - p_z)
    assert (1, 0) in ap_pairs
    assert ap_pairs[(1, 0)] == pytest.approx(expected_pp, rel=1e-6)
    assert ap_overall == pytest.approx(ap_pairs[(1, 0)], rel=1e-6)


def test_compute_attention_plasticity_respects_bucket_window_limits():
    p_q = np.array([0.0, 1.0, 2.0])
    b_q = np.array([0, 1, 2])
    X_q_rot = np.array([[0.0], [1.0], [2.0]])
    alpha_rot = np.array([0.0])
    beta_rot = np.array([1.0])
    resid_var = np.array([1.0])

    p_k = np.array([0.0, 0.1, 1.0, 1.1])
    b_k = np.array([0, 0, 1, 1])
    X_k_rot = np.array([[1.0], [0.5], [2.0], [1.5]])
    example_ids = np.array([0, 0, 1, 1])

    ap_overall, ap_pairs = compute_attention_plasticity(
        p_q=p_q,
        b_q=b_q,
        X_q_rot=X_q_rot,
        alpha_rot=alpha_rot,
        beta_rot=beta_rot,
        resid_var=resid_var,
        p_k=p_k,
        b_k=b_k,
        example_ids_k=example_ids,
        X_k_rot=X_k_rot,
        num_pairs_per_bucket=1,
        seed=0,
        bucket_window_limits=None,
    )

    assert (2, 0) in ap_pairs
    assert not math.isnan(ap_pairs[(2, 0)])
    assert (2, 1) in ap_pairs
    assert not math.isnan(ap_pairs[(2, 1)])

    _, constrained = compute_attention_plasticity(
        p_q=p_q,
        b_q=b_q,
        X_q_rot=X_q_rot,
        alpha_rot=alpha_rot,
        beta_rot=beta_rot,
        resid_var=resid_var,
        p_k=p_k,
        b_k=b_k,
        example_ids_k=example_ids,
        X_k_rot=X_k_rot,
        num_pairs_per_bucket=1,
        seed=0,
        bucket_window_limits={2: 1},
    )

    assert (2, 0) not in constrained
    assert (2, 1) in constrained
    assert not math.isnan(constrained[(2, 1)])


def test_compute_attention_plasticity_enforces_same_example_pairs():
    p_q = np.array([1.0])
    b_q = np.array([1])
    X_q_rot = np.array([[0.0]])
    alpha_rot = np.array([1.0])
    beta_rot = np.array([0.0])
    resid_var = np.array([1.0])

    p_k = np.zeros(3)
    b_k = np.array([0, 0, 0])
    X_k_rot = np.array([[0.0], [1.0], [100.0]])
    example_ids = np.array([0, 0, 1])

    ap_overall, ap_pairs = compute_attention_plasticity(
        p_q=p_q,
        b_q=b_q,
        X_q_rot=X_q_rot,
        alpha_rot=alpha_rot,
        beta_rot=beta_rot,
        resid_var=resid_var,
        p_k=p_k,
        b_k=b_k,
        example_ids_k=example_ids,
        X_k_rot=X_k_rot,
        num_pairs_per_bucket=1,
        seed=0,
    )

    assert (1, 0) in ap_pairs
    p_expected = 0.5 * (1 + math.erf(-1.0 / math.sqrt(2)))
    expected_pp = 4 * p_expected * (1 - p_expected)
    assert ap_pairs[(1, 0)] == pytest.approx(expected_pp, rel=1e-6)
    assert ap_overall == pytest.approx(expected_pp, rel=1e-6)


def test_analyze_head_with_dummy_data(monkeypatch):
    positions_q = np.array([0.0, 1.0, 2.0, 3.0])
    buckets_q = np.array([0, 0, 1, 1])
    vectors_q = [np.array([p, -p], dtype=np.float64) for p in positions_q]
    df_q = make_df(vectors_q, buckets_q, positions_q)

    positions_k = np.array([0.0, 0.5, 1.5, 2.5])
    buckets_k = np.array([0, 0, 1, 1])
    vectors_k = [np.array([p + 0.1, -(p + 0.1)], dtype=np.float64) for p in positions_k]
    df_k = make_df(vectors_k, buckets_k, positions_k, example_ids=[0, 0, 1, 1])

    model_dir = "dummy_model"
    dataset_store = {
        f"{model_dir}/l00h00q": DummyDataset(df_q),
        f"{model_dir}/l00h00k": DummyDataset(df_k),
    }

    dataset_name = "custom/sniffed-qk"

    def fake_hf_hub_url(repo_id, filename, repo_type):
        assert repo_id == dataset_name
        assert repo_type == "dataset"
        return filename

    def fake_load_dataset(name, *, data_files=None, split=None):
        assert name == "parquet"
        assert data_files is not None
        key = data_files.rsplit("/data.parquet", 1)[0]
        return dataset_store[key]

    monkeypatch.setattr(head_analysis, "hf_hub_url", fake_hf_hub_url)
    monkeypatch.setattr(head_analysis, "load_dataset", fake_load_dataset)

    row, bucket_rows, component_rows = analyze_head(
        layer=0,
        q_head=0,
        k_head=0,
        model_dir=model_dir,
        max_tokens_per_head=10,
        normality_max_dims=2,
        p_alpha=0.05,
        seed=0,
        dataset_name=dataset_name,
    )

    assert row["layer"] == 0
    assert row["q_head"] == 0
    assert row["k_head"] == 0
    assert row["n_q_tokens"] == 4
    assert row["d_model"] == 2
    assert row["q_R2_pos"] == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= row["ap_overall"] <= 1.0
    assert bucket_rows
    assert all(entry["q_bucket"] > 0 for entry in bucket_rows)
    assert {(entry["q_bucket"], entry["k_bucket"]) for entry in bucket_rows} == {(1, 0)}
    for entry in bucket_rows:
        assert entry["layer"] == 0
        assert entry["q_head"] == 0
        assert entry["k_head"] == 0
    assert len(component_rows) == 2
    weights = [entry["component_weight"] for entry in component_rows]
    assert pytest.approx(sum(weights), rel=1e-6) == 1.0


def test_analyze_head_respects_sliding_window_column(monkeypatch):
    positions_q = np.array([0.0, 1.0, 2.0])
    buckets_q = np.array([0, 1, 2])
    vectors_q = [np.array([p, -p], dtype=np.float64) for p in positions_q]

    positions_k = np.array([0.0, 0.1, 1.0, 1.1])
    buckets_k = np.array([0, 0, 1, 1])
    vectors_k = [
        np.array([1.0, -1.0]),
        np.array([1.5, -1.5]),
        np.array([2.0, -2.0]),
        np.array([2.5, -2.5]),
    ]
    df_k = make_df(vectors_k, buckets_k, positions_k, example_ids=[0, 0, 1, 1])

    dataset_name = "custom/sniffed-qk"
    model_dir = "dummy_model"

    def run_analysis(sliding_values):
        df_q = make_df(vectors_q, buckets_q, positions_q, sliding_window=sliding_values)
        dataset_store = {
            f"{model_dir}/l00h00q": DummyDataset(df_q),
            f"{model_dir}/l00h00k": DummyDataset(df_k),
        }

        def fake_hf_hub_url(repo_id, filename, repo_type):
            assert repo_id == dataset_name
            assert repo_type == "dataset"
            return filename

        def fake_load_dataset(name, *, data_files=None, split=None):
            assert name == "parquet"
            assert data_files is not None
            key = data_files.rsplit("/data.parquet", 1)[0]
            return dataset_store[key]

        monkeypatch.setattr(head_analysis, "hf_hub_url", fake_hf_hub_url)
        monkeypatch.setattr(head_analysis, "load_dataset", fake_load_dataset)

        _, bucket_rows, component_rows = analyze_head(
            layer=0,
            q_head=0,
            k_head=0,
            model_dir=model_dir,
            max_tokens_per_head=10,
            normality_max_dims=2,
            p_alpha=0.05,
            seed=0,
            dataset_name=dataset_name,
        )
        assert len(component_rows) == len(vectors_q[0])
        weight_sum = sum(entry["component_weight"] for entry in component_rows)
        assert pytest.approx(weight_sum, rel=1e-6) == 1.0

        return {
            (entry["q_bucket"], entry["k_bucket"]): entry["ap_bucket"]
            for entry in bucket_rows
        }

    full_map = run_analysis([0, 0, 0])
    assert (2, 0) in full_map
    assert not math.isnan(full_map[(2, 0)])

    window_map = run_analysis([1, 1, 1])
    assert (2, 0) not in window_map


def test_analyze_head_rejects_mixed_sliding_window(monkeypatch):
    positions_q = np.array([0.0, 1.0])
    buckets_q = np.array([0, 1])
    vectors_q = [np.array([p, -p], dtype=np.float64) for p in positions_q]
    df_q = make_df(vectors_q, buckets_q, positions_q, sliding_window=[0, 1])

    positions_k = np.array([0.0, 0.1])
    buckets_k = np.array([0, 0])
    vectors_k = [np.array([1.0, -1.0]), np.array([-1.0, 1.0])]
    df_k = make_df(vectors_k, buckets_k, positions_k, example_ids=[0, 0])

    dataset_name = "custom/sniffed-qk"
    model_dir = "dummy_model"
    dataset_store = {
        f"{model_dir}/l00h00q": DummyDataset(df_q),
        f"{model_dir}/l00h00k": DummyDataset(df_k),
    }

    def fake_hf_hub_url(repo_id, filename, repo_type):
        assert repo_id == dataset_name
        assert repo_type == "dataset"
        return filename

    def fake_load_dataset(name, *, data_files=None, split=None):
        assert name == "parquet"
        assert data_files is not None
        key = data_files.rsplit("/data.parquet", 1)[0]
        return dataset_store[key]

    monkeypatch.setattr(head_analysis, "hf_hub_url", fake_hf_hub_url)
    monkeypatch.setattr(head_analysis, "load_dataset", fake_load_dataset)

    with pytest.raises(ValueError):
        analyze_head(
            layer=0,
            q_head=0,
            k_head=0,
            model_dir=model_dir,
            max_tokens_per_head=10,
            normality_max_dims=2,
            p_alpha=0.05,
            seed=0,
            dataset_name=dataset_name,
        )


def test_analyze_head_uses_local_dataset(monkeypatch, tmp_path):
    positions_q = np.array([0.0, 1.0])
    buckets_q = np.array([0, 1])
    vectors_q = [np.array([p, -p], dtype=np.float64) for p in positions_q]
    df_q = make_df(vectors_q, buckets_q, positions_q)

    positions_k = np.array([0.0, 0.5])
    buckets_k = np.array([0, 0])
    vectors_k = [np.array([1.0, -1.0]), np.array([2.0, -2.0])]
    df_k = make_df(vectors_k, buckets_k, positions_k, example_ids=[0, 0])

    dataset_root = tmp_path / "snapshot"
    model_dir = "dummy_model"
    q_path = dataset_root / model_dir / "l00h00q" / "data.parquet"
    k_path = dataset_root / model_dir / "l00h00k" / "data.parquet"
    q_path.parent.mkdir(parents=True, exist_ok=True)
    k_path.parent.mkdir(parents=True, exist_ok=True)
    q_path.write_text("")
    k_path.write_text("")

    dataset_store = {
        str(q_path): DummyDataset(df_q),
        str(k_path): DummyDataset(df_k),
    }

    def fake_load_dataset(name, *, data_files=None, split=None):
        assert name == "parquet"
        assert data_files in dataset_store
        return dataset_store[data_files]

    def fail_hf_hub_url(*args, **kwargs):
        raise AssertionError("hf_hub_url should not be called when dataset_local_root is set")

    monkeypatch.setattr(head_analysis, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(head_analysis, "hf_hub_url", fail_hf_hub_url)

    row, bucket_rows, component_rows = analyze_head(
        layer=0,
        q_head=0,
        k_head=0,
        model_dir=model_dir,
        max_tokens_per_head=10,
        normality_max_dims=2,
        p_alpha=0.05,
        seed=0,
        dataset_name="custom/sniffed-qk",
        dataset_local_root=str(dataset_root),
    )

    assert row["n_q_tokens"] == 2
    assert component_rows
    assert pytest.approx(sum(entry["component_weight"] for entry in component_rows), rel=1e-6) == 1.0
