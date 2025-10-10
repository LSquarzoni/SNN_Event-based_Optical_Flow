#include <torch/script.h>
#include <ATen/ATen.h>
#include <tuple>
#include <unordered_map>
#include <mutex>

std::tuple<torch::Tensor, torch::Tensor> LIF(
    torch::Tensor input,         // [N, C, H, W]
    torch::Tensor mem,           // [N, C, H, W]
    torch::Tensor beta,          // [C]
    torch::Tensor threshold      // [C]
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

static auto registry = torch::RegisterOperators("SNN_implementation::LIF", &LIF);

// Stateful LIF with internal persistent membrane state per state_id
// Signature:
//   input:     [N, C, H, W]
//   mem_in:    optional [N, C, H, W] (when provided, overrides internal state)
//   beta:      [C]
//   threshold: [C]
//   state_id:  int64 identifier for the internal state slot
// Returns:
//   spike:     [N, C, H, W]

namespace {
// Global state map guarded by a mutex. One state tensor per state_id.
std::unordered_map<int64_t, torch::Tensor> g_state_map;
std::mutex g_state_mutex;
}

torch::Tensor LIF_stateful(
    const torch::Tensor& input,
    const c10::optional<torch::Tensor>& mem_in,
    const torch::Tensor& beta,
    const torch::Tensor& threshold,
    int64_t state_id) {

    TORCH_CHECK(input.dim() == 4, "input must be NCHW");
    TORCH_CHECK(beta.dim() == 1 && threshold.dim() == 1, "beta and threshold must be [C]");

    auto sizes = input.sizes();
    int64_t N = sizes[0];
    int64_t C = sizes[1];
    int64_t H = sizes[2];
    int64_t W = sizes[3];

    TORCH_CHECK(beta.size(0) == C && threshold.size(0) == C, "beta and threshold size must match C");

    // Acquire or initialize internal membrane state
    torch::Tensor mem;
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        auto it = g_state_map.find(state_id);
        if (mem_in.has_value()) {
            // Use provided mem_in to initialize/override state
            TORCH_CHECK(mem_in.value().sizes() == input.sizes(), "mem_in must match input shape");
            mem = mem_in.value().to(input.options());
            g_state_map[state_id] = mem.clone();
        } else if (it == g_state_map.end()) {
            // First use: initialize to zeros
            mem = torch::zeros_like(input);
            g_state_map[state_id] = mem.clone();
        } else {
            // Reuse existing state; resize if shape changed
            mem = it->second;
            if (!mem.defined() || mem.sizes() != input.sizes() || mem.dtype() != input.dtype() || mem.device() != input.device()) {
                mem = torch::zeros_like(input);
                g_state_map[state_id] = mem.clone();
            }
        }
    }

    // Broadcast beta and threshold to NCHW
    auto beta_b = beta.view({1, C, 1, 1}).to(input.options());
    auto thr_b = threshold.view({1, C, 1, 1}).to(input.options());

    // Compute updated membrane and spike
    auto updated_mem = beta_b * mem + input;
    auto spike = updated_mem.ge(thr_b).to(input.dtype());
    auto mem_out = torch::where(spike.to(torch::kBool), torch::zeros_like(updated_mem), updated_mem);

    // Persist updated membrane state
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        g_state_map[state_id] = mem_out.detach();
    }

    return spike;
}

static auto registry_stateful = torch::RegisterOperators("SNN_implementation::LIF_stateful", &LIF_stateful);