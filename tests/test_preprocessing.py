import numpy as np
import pandas as pd

from src.data_cleaning import preprocess_data


def test_mixed_feature_preprocessing_handles_missing_values():
    frame = pd.DataFrame({"amount": [10.0, np.nan, 30.0, 40.0], "sector": ["A", "B", None, "A"]})
    matrix, preprocessor, selected = preprocess_data(frame, ["amount", "sector"])
    dense = matrix.toarray() if hasattr(matrix, "toarray") else matrix
    assert selected == ["amount", "sector"]
    assert dense.shape == (4, 3)
    assert np.isfinite(dense).all()
    assert preprocessor is not None