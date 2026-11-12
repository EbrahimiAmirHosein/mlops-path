import numpy as np

from train import build_model, load_data


def test_split_shapes():
    X_tr, X_te, y_tr, y_te = load_data()
    assert X_tr.shape[0] == y_tr.shape[0]
    assert X_te.shape[0] == y_te.shape[0]
    assert X_tr.shape[1] == 30
    assert X_te.shape[1] == 30



def test_predict_proba_shape():
    X_tr, X_te, y_tr, _y_te = load_data()
    model = build_model().fit(X_tr[:100], y_tr[:100])
    proba = model.predict_proba(X_te)
    assert proba.shape == (len(X_te) , 2)
    assert np.allclose(proba.sum(axis=1), 1)



def test_reproducible():
    first = load_data(seed=42)
    second = load_data(seed=42)
    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert np.array_equal(a, b)