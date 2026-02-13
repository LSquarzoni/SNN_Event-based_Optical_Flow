#include <torch/script.h>
#include <ATen/ATen.h>
#include <tuple>
#include <unordered_map>
#include <mutex>
#include <torch/library.h>

std::tuple<torch::Tensor, torch::Tensor> LIF(
    const torch::Tensor& input,         // [N, C, H, W]
    const torch::Tensor& mem,           // [N, C, H, W]
    const torch::Tensor& beta,          // [C]
    const torch::Tensor& threshold      // [C]
) {
    // Get raw pointers
    float* input_data = input.data_ptr<float>();
    float* mem_data = mem.data_ptr<float>();
    float* beta_data = beta.data_ptr<float>();
    float* threshold_data = threshold.data_ptr<float>();

    // Get dimensions
    auto sizes = input.sizes();
    int64_t N = sizes[0];
    int64_t C = sizes[1];
    int64_t H = sizes[2];
    int64_t W = sizes[3];

    // Output tensors
    torch::Tensor spike = torch::zeros_like(input);
    torch::Tensor mem_out = torch::zeros_like(input);
    float* spike_data = spike.data_ptr<float>();
    float* mem_out_data = mem_out.data_ptr<float>();

    // LIF computation
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            float beta_val = beta_data[c];
            float threshold_val = threshold_data[c];
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    int64_t idx = ((n * C + c) * H + h) * W + w;
                    float updated_mem = beta_val * mem_data[idx] + input_data[idx];
                    if (updated_mem >= threshold_val) {
                        spike_data[idx] = 1.0f;
                        mem_out_data[idx] = 0.0f; // reset to zero after spike
                    } else {
                        spike_data[idx] = 0.0f;
                        mem_out_data[idx] = updated_mem;
                    }
                }
            }
        }
    }

    // Return as a tuple: (spike, mem_out), each with shape [N, C, H, W]
    return std::make_tuple(spike, mem_out);
}

// Meta kernel
std::tuple<torch::Tensor, torch::Tensor> LIF_meta(
    const torch::Tensor& input,
    const torch::Tensor& mem,
    const torch::Tensor& beta,
    const torch::Tensor& threshold
) {
    auto out1 = at::empty_like(input, input.options().device(at::kMeta));
    auto out2 = at::empty_like(input, input.options().device(at::kMeta));
    return std::make_tuple(out1, out2);
}

// Register the operator schema
TORCH_LIBRARY(SNN_implementation, m) {
    m.def("LIF(Tensor input, Tensor mem, Tensor beta, Tensor threshold) -> (Tensor, Tensor)");
}

// Register CPU kernel
TORCH_LIBRARY_IMPL(SNN_implementation, CPU, m) {
    m.impl("LIF", LIF);
}

// Register meta kernel
TORCH_LIBRARY_IMPL(SNN_implementation, Meta, m) {
    m.impl("LIF", LIF_meta);
}