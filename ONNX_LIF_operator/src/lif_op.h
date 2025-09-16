#include <torch/script.h>
#include <torch/extension.h>

class LIFOperator {
public:
    LIFOperator(float beta, float threshold);
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor input, torch::Tensor mem);
    std::tuple<torch::Tensor, torch::Tensor> backward(torch::Tensor grad_output);

private:
    float beta_;
    float threshold_;
    torch::Tensor mem_;
};