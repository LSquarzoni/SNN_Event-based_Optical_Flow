import unittest
import torch
from src.python.lif_op import LIFOperator  # Adjust the import based on your actual implementation

class TestLIFOperator(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.input_size = 3
        self.mem = torch.zeros(self.batch_size, self.input_size)
        self.current_input = torch.tensor([[0.5, 0.2, 0.1], [0.3, 0.4, 0.6]], dtype=torch.float32)
        self.lif_operator = LIFOperator(beta=0.5, threshold=1.0)

    def test_forward(self):
        spikes, mem_out = self.lif_operator.forward(self.current_input, self.mem)
        self.assertEqual(spikes.shape, (self.batch_size, self.input_size))
        self.assertEqual(mem_out.shape, (self.batch_size, self.input_size))

    def test_spike_generation(self):
        self.mem = torch.tensor([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]], dtype=torch.float32)
        spikes, _ = self.lif_operator.forward(self.current_input, self.mem)
        self.assertTrue(torch.all(spikes <= 1.0))
        self.assertTrue(torch.all(spikes >= 0.0))

    def test_membrane_potential_update(self):
        _, mem_out = self.lif_operator.forward(self.current_input, self.mem)
        self.assertTrue(torch.all(mem_out >= 0.0))

if __name__ == '__main__':
    unittest.main()