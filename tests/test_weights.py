"""Numerical and artifact-integrity checks; no model downloads or inference."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'weights.py'
spec = importlib.util.spec_from_file_location('weights', SCRIPT)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


class WeightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'base.safetensors'
        self.direction = self.root / 'direction.json'
        self.direction.write_text('[1, 2, -1]')
        self.tensor = torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]])
        save_file({'down.weight': self.tensor, 'other': torch.tensor([2., 3.])},
                  str(self.source), metadata={'format': 'pt', 'test': 'keep'})
        self.original_hash = w.sha256(self.source)

    def test_projection_geometry_and_unselected_columns(self):
        out = w.project_tensor(self.tensor, [1, 2, -1], [1, 3], 1.0)
        u = torch.tensor([1., 2., -1.], dtype=torch.float64)
        u /= u.norm()
        self.assertTrue(torch.allclose(u @ out[:, [1, 3]].double(), torch.zeros(2, dtype=torch.float64), atol=1e-6))
        self.assertTrue(torch.equal(out[:, [0, 2]], self.tensor[:, [0, 2]]))
        self.assertTrue(torch.equal(self.tensor, torch.arange(1., 13.).reshape(3, 4)))

    def test_write_diff_metadata_hashes_and_rollback(self):
        target = self.root / 'candidate.safetensors'
        manifest = w.project_file(self.source, 'down.weight', self.direction, [1, 3], output=target)
        data = load_file(str(target))
        self.assertTrue(torch.equal(data['other'], torch.tensor([2., 3.])))
        self.assertEqual(w.sha256(self.source), self.original_hash)
        self.assertEqual(manifest['output_sha256'], w.sha256(target))
        self.assertEqual(manifest['source_sha256'], self.original_hash)
        self.assertEqual(manifest, json.loads(Path(str(target) + '.manifest.json').read_text()))
        with safe_open(target, framework='pt') as f:
            self.assertEqual(f.metadata(), {'format': 'pt', 'test': 'keep'})
        diff = w.compare_files(self.source, target)
        self.assertEqual(diff['changed_elements'], manifest['changed_elements'])
        self.assertEqual(diff['total_stored_elements'], 14)
        self.assertEqual(len(diff['changed_tensors']), 1)
        self.assertEqual(w.compare_files(self.source, self.source)['changed_elements'], 0)

    def test_preview_writes_nothing(self):
        before = set(self.root.iterdir())
        result = w.project_file(self.source, 'down.weight', self.direction, [0])
        self.assertFalse(result['written'])
        self.assertIsNone(result['output'])
        self.assertEqual(before, set(self.root.iterdir()))
        self.assertEqual(w.sha256(self.source), self.original_hash)

    def test_low_precision_rounding_is_measured(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                a = torch.tensor([[1., 2.], [3., 4.]], dtype=dtype)
                b = w.project_tensor(a, [1, 0], [0], 1.0)
                self.assertEqual(b.dtype, dtype)
                self.assertEqual(b[0, 0].item(), 0)
                self.assertEqual(w.tensor_difference(a, b)['changed_elements'], 1)
                c = w.project_tensor(a, [1, 0], [0], 1e-20)
                self.assertEqual(w.tensor_difference(a, c)['changed_elements'], 0)

    def test_invalid_projection_inputs(self):
        cases = [([0, 0, 0], [0], 1), ([1, 2], [0], 1),
                 ([1, float('nan'), 0], [0], 1), ([1, 0, 0], [], 1),
                 ([1, 0, 0], [4], 1), ([1, 0, 0], [-1], 1),
                 ([1, 0, 0], [0, 0], 1), ([1, 0, 0], [True], 1),
                 ([1, 0, 0], [0], 0), ([1, 0, 0], [0], float('inf'))]
        for d, columns, alpha in cases:
            with self.subTest(d=d, columns=columns, alpha=alpha), self.assertRaises((ValueError, TypeError)):
                w.project_tensor(self.tensor, d, columns, alpha)
        with self.assertRaises(ValueError):
            w.project_tensor(self.tensor.to(torch.int32), [1, 0, 0], [0], 1)
        bad = self.tensor.clone()
        bad[0, 0] = float('nan')
        with self.assertRaises(ValueError):
            w.project_tensor(bad, [1, 0, 0], [0], 1)

    def test_extreme_direction_normalization(self):
        a = w.project_tensor(self.tensor, [1e300, 0, 0], [0], 1)
        b = w.project_tensor(self.tensor, [1e-300, 0, 0], [0], 1)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(a[0, 0], 0)

    def test_no_overwrite_or_partial_candidate_on_failure(self):
        target = self.root / 'candidate.safetensors'
        target.write_bytes(b'keep')
        for output in (target, self.source):
            with self.assertRaises(ValueError):
                w.project_file(self.source, 'down.weight', self.direction, [0], output=output)
        self.assertEqual(target.read_bytes(), b'keep')
        fresh = self.root / 'fresh.safetensors'
        with self.assertRaises(ValueError):
            w.project_file(self.source, 'missing', self.direction, [0], output=fresh)
        self.assertFalse(fresh.exists())
        self.assertFalse(Path(str(fresh) + '.manifest.json').exists())
        self.assertEqual(w.sha256(self.source), self.original_hash)

    def test_schema_mismatch_and_nonfinite_diff(self):
        for data in ({'different': self.tensor},
                     {'down.weight': self.tensor.double(), 'other': torch.tensor([2., 3.])},
                     {'down.weight': self.tensor, 'other': torch.tensor([float('nan'), 3.])}):
            target = self.root / 'bad.safetensors'
            save_file(data, str(target))
            with self.assertRaises(ValueError):
                w.compare_files(self.source, target)

    def test_inventory_filter_keeps_full_denominator_and_reports_nonfinite(self):
        target = self.root / 'nonfinite.safetensors'
        save_file({'bad': torch.tensor([float('inf'), float('nan'), 2.]),
                   'other': torch.tensor([1.])}, str(target))
        report = w.inspect_file(target, 'bad')
        self.assertEqual(report['total_stored_elements'], 4)
        self.assertEqual(len(report['tensors']), 1)
        self.assertEqual(report['tensors'][0]['nonfinite'], 2)
        self.assertEqual(report['tensors'][0]['finite_l2'], 2)

    def test_cli_preview_is_json_and_read_only(self):
        result = subprocess.run([sys.executable, str(SCRIPT), 'project', str(self.source),
                                 '--tensor', 'down.weight', '--direction', str(self.direction),
                                 '--columns', '1,3'], capture_output=True, text=True, check=True)
        self.assertFalse(json.loads(result.stdout)['written'])
        self.assertEqual(w.sha256(self.source), self.original_hash)


if __name__ == '__main__':
    unittest.main()
