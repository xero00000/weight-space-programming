"""Finite-domain verification of independently constructed attention/MLP circuits."""
import importlib.util
import itertools
from pathlib import Path
import unittest

import torch

SCRIPT = Path(__file__).resolve().parents[1] / 'examples' / 'transformer_circuits.py'
spec = importlib.util.spec_from_file_location('transformer_circuits', SCRIPT)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class CircuitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_parity_exhaustive_nonempty_domain(self):
        strings = [''.join(bits) for n in range(1, 11)
                   for bits in itertools.product('01', repeat=n)]
        tokens = torch.tensor([c.encode_bits(s) for s in strings], dtype=torch.long)
        # The independent oracle is only used by the evaluator, never by the model.
        labels = torch.tensor([s.count('1') % 2 for s in strings])
        self.assertEqual(len(strings), 2046)
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                model = c.ParityCircuit().to(dtype=dtype)
                logits = model(tokens)
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue(torch.equal(logits.argmax(dim=-1), labels))
                correct = logits.gather(1, labels[:, None]).squeeze(1)
                opposite = logits.gather(1, (1 - labels)[:, None]).squeeze(1)
                self.assertGreater(float((correct - opposite).min()), 0.99)

    def test_router_exhaustive_inputs_and_source_positions(self):
        tokens = torch.tensor(list(itertools.product(range(4), repeat=4)), dtype=torch.long)
        self.assertEqual(len(tokens), 256)
        for dtype, position in itertools.product((torch.float32, torch.float64), range(4)):
            with self.subTest(dtype=dtype, position=position):
                logits = c.PositionRouter(position).to(dtype=dtype)(tokens)
                labels = tokens[:, position]
                self.assertTrue(torch.equal(logits.argmax(dim=-1), labels))
                expected_mass = 1 / (1 + 3 * torch.exp(torch.tensor(-16., dtype=dtype)))
                correct = logits.gather(1, labels[:, None]).squeeze(1)
                self.assertGreaterEqual(float(correct.min()), float(expected_mass) - 1e-7)
                self.assertTrue(torch.allclose(logits.sum(dim=-1), torch.ones(256, dtype=dtype)))

    def test_state_is_frozen_and_inference_preserves_weights(self):
        for model, tokens in ((c.ParityCircuit(), [c.encode_bits('101')]),
                              (c.PositionRouter(2), [[0, 1, 2, 3]])):
            before = {k: v.clone() for k, v in model.state_dict().items()}
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            model(torch.tensor(tokens, dtype=torch.long))
            self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))
            summary = c.inventory(model)
            self.assertGreater(summary['parameter_elements'], 0)
            self.assertGreater(summary['buffer_elements'], 0)
            self.assertGreater(summary['registered_tensor_bytes'], summary['parameter_elements'])

    def test_invalid_input_contracts(self):
        for bits in ('', 'x', '1' * 11):
            with self.assertRaises(ValueError):
                c.encode_bits(bits)
        for position in (-1, 4, True):
            with self.assertRaises(ValueError):
                c.PositionRouter(position)
        for model, tokens in ((c.PositionRouter(0), torch.zeros(1, 3, dtype=torch.long)),
                              (c.PositionRouter(0), torch.zeros(1, 4)),
                              (c.PositionRouter(0), torch.tensor([[0, 1, 2, 4]])),
                              (c.ParityCircuit(), torch.zeros(1, 11, dtype=torch.long))):
            with self.assertRaises(ValueError):
                model(tokens)


if __name__ == '__main__':
    unittest.main()
