#include <onnxruntime_cxx_api.h>
#include <vector>
#include <cassert>

using namespace ONNX_NAMESPACE;

class LIFOperator : public Ort::CustomOpBase<LIFOperator, Ort::Kernel> {
public:
    void Compute(OrtKernelContext* context) {
        // Get input tensors
        const OrtValue* input_tensor = ort_api->KernelContext_GetInput(context, 0);
        const OrtValue* mem_tensor = ort_api->KernelContext_GetInput(context, 1);
        const OrtValue* beta_tensor = ort_api->KernelContext_GetInput(context, 2);
        const OrtValue* threshold_tensor = ort_api->KernelContext_GetInput(context, 3);

        // Get input shapes
        OrtTensorDimensions input_dims(ort_api, input_tensor);
        assert(input_dims.size() == 4); // [batch, channels, height, width]
        size_t num_batches = input_dims[0];
        size_t num_channels = input_dims[1];
        size_t height = input_dims[2];
        size_t width = input_dims[3];
        size_t num_elements = num_batches * num_channels * height * width;

        // Get data pointers
        const float* input_data = ort_api->GetTensorData<float>(input_tensor);
        const float* mem_data = ort_api->GetTensorData<float>(mem_tensor);
        const float* beta_data = ort_api->GetTensorData<float>(beta_tensor);       // shape: [channels]
        const float* threshold_data = ort_api->GetTensorData<float>(threshold_tensor); // shape: [channels]

        // Get output tensors
        OrtValue* spike_tensor = ort_api->KernelContext_GetOutput(context, 0, input_dims.data(), input_dims.size());
        OrtValue* mem_out_tensor = ort_api->KernelContext_GetOutput(context, 1, input_dims.data(), input_dims.size());
        float* spike_data = ort_api->GetTensorMutableData<float>(spike_tensor);
        float* mem_out_data = ort_api->GetTensorMutableData<float>(mem_out_tensor);

        // LIF computation with reset-to-zero and per-channel beta/threshold
        // Store updated membrane voltage as the state of the LIF layer
        for (size_t b = 0; b < num_batches; ++b) {
            for (size_t c = 0; c < num_channels; ++c) {
                float beta = beta_data[c];
                float threshold = threshold_data[c];
                for (size_t h = 0; h < height; ++h) {
                    for (size_t w = 0; w < width; ++w) {
                        size_t idx = ((b * num_channels + c) * height + h) * width + w;
                        float updated_mem = beta * mem_data[idx] + input_data[idx];
                        if (updated_mem >= threshold) {
                            spike_data[idx] = 1.0f;
                            mem_out_data[idx] = 0.0f; // reset to zero after spike
                        } else {
                            spike_data[idx] = 0.0f;
                            mem_out_data[idx] = updated_mem; // store updated membrane voltage
                        }
                    }
                }
            }
        }
    }
};

// Register the operator
ORT_API_STATUS* CreateLIFOperator(Ort::CustomOpApi api, const OrtKernelInfo* info) {
    return new LIFOperator();
}