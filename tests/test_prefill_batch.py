import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('batch_test_impl', ROOT/'patches/batching/switchyard_batch.py')
module = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {
    'model_router_toolkit.prefill.transforms': SimpleNamespace(build_trunk_features=None),
    'model_router_toolkit.prefill.trunk': SimpleNamespace(predict_proba=None),
}):
    spec.loader.exec_module(module)


class BatchGuardTests(unittest.TestCase):
    @patch.dict(os.environ, {'SWITCHYARD_PREFILL_TOLERANCE':'0.195','SWITCHYARD_PREFILL_NUMERIC_GUARD':'0.06'})
    def test_rechecks_only_boundary_and_preserves_order(self):
        calls=[]
        def score(prompt):
            calls.append(prompt)
            return SimpleNamespace(confidences=[.5,.8])
        scorer=SimpleNamespace(score=score)
        raw=np.array([[.8,.85],[.61,.8],[.1,.9]],dtype=np.float32)
        with patch.object(module,'score_batch',return_value=(raw, {})):
            out=module.forward(scorer,['easy','boundary','hard'],16,2048)
        self.assertEqual(calls,['boundary'])
        np.testing.assert_allclose(out,[[.8,.85],[.5,.8],[.1,.9]])

    def test_single_request_uses_original_score(self):
        scorer=SimpleNamespace(score=lambda _:SimpleNamespace(confidences=[.2,.8]))
        with patch.object(module,'score_batch',side_effect=AssertionError('should not batch')):
            np.testing.assert_allclose(module.forward(scorer,['one'],16,2048),[[.2,.8]])

    @patch.dict(os.environ, {'SWITCHYARD_PREFILL_TOLERANCE':'0.195','SWITCHYARD_PREFILL_NUMERIC_GUARD':'-1'})
    def test_invalid_guard_rejected(self):
        with patch.object(module,'score_batch',return_value=(np.array([[.5,.8],[.5,.8]]), {})):
            with self.assertRaises(ValueError):module.forward(None,['a','b'],16,2048)


if __name__=='__main__':unittest.main()
