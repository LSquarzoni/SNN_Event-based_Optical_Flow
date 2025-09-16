#include <onnxruntime_cxx_api.h>
#include <torch/script.h>

torch::Tensor lif_leaky(
    torch::Tensor input,         // [N, C, H, W]
    torch::Tensor mem,           // [N, C, H, W]
    torch::Tensor beta,          // [C]
    torch::Tensor threshold      // [C]
) {
    // Get raw pointers
    float* input_data = input.data<float>();
    float* mem_data = mem.data<float>();
    float* beta_data = beta.data<float>();
    float* threshold_data = threshold.data<float>();

    // Get dimensions
    auto sizes = input.sizes();
    int64_t N = sizes[0];
    int64_t C = sizes[1];
    int64_t H = sizes[2];
    int64_t W = sizes[3];
    int64_t numel = N * C * H * W;

    // Output tensors
    torch::Tensor spike = torch::zeros_like(input);
    torch::Tensor mem_out = torch::zeros_like(input);
    float* spike_data = spike.data<float>();
    float* mem_out_data = mem_out.data<float>();

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

    // You can return a tuple or just one tensor depending on your needs
    // Example: return torch::stack({spike, mem_out}, 0);
    // Or, if you want to return both:
    return torch::cat({spike.unsqueeze(0), mem_out.unsqueeze(0)}, 0); // shape [2, N, C, H, W]
}

static auto registry = torch::RegisterOperators("mynamespace::lif_leaky", &lif_leaky);