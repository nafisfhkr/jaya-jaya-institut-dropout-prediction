from pathlib import Path
import sys
import unittest

import joblib

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app import build_input_frame, validate_model_bundle


class ModelBundleValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = joblib.load(PROJECT_DIR / 'model' / 'dropout_pipeline.joblib')

    def test_bundle_has_valid_category_values(self) -> None:
        validate_model_bundle(self.bundle)

    def test_invalid_category_is_rejected(self) -> None:
        values = dict(self.bundle['feature_defaults'])
        values['Marital_status'] = 999

        with self.assertRaises(ValueError):
            build_input_frame(self.bundle, values)


if __name__ == '__main__':
    unittest.main()
