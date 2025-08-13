#include "ActivationFunction.hpp"

#include "ExtendedExpection.hpp"
#include "GELU.hpp"
#include "Gaussian.hpp"
#include "Identity.hpp"
#include "ImprovedSigmoid.hpp"
#include "LeakyReLU.hpp"
#include "ReLU.hpp"
#include "Sigmoid.hpp"
#include "Tanh.hpp"

namespace snn::internal
{
std::vector<std::shared_ptr<ActivationFunction>> ActivationFunction::activationFunctions = initialize();

inline auto ActivationFunction::initialize() -> std::vector<std::shared_ptr<ActivationFunction>>
{
    std::vector<std::shared_ptr<ActivationFunction>> activations;
    constexpr size_t numberOfActivationFunctions = 8;
    activations.reserve(numberOfActivationFunctions);
    activations.emplace_back(std::make_shared<Sigmoid>());
    activations.emplace_back(std::make_shared<ImprovedSigmoid>());
    activations.emplace_back(std::make_shared<Tanh>());
    activations.emplace_back(std::make_shared<RectifiedLinearUnit>());
    activations.emplace_back(std::make_shared<GaussianErrorLinearUnit>());
    activations.emplace_back(std::make_shared<Gaussian>());
    activations.emplace_back(std::make_shared<Identity>());
    activations.emplace_back(std::make_shared<LeakyRectifiedLinearUnit>());
    return activations;
}

ActivationFunction::ActivationFunction(float min, float max)
    : min(min),
      max(max)
{
}

auto ActivationFunction::get(activation type) -> std::shared_ptr<ActivationFunction>
{
    const auto index = static_cast<uint8_t>(type);
    if (index > activationFunctions.size() - 1)
    {
        throw NotImplementedException("activation");
    }
    return activationFunctions[index];
}

auto ActivationFunction::operator==(const ActivationFunction& activationFunction) const -> bool
{
    return this->getType() == activationFunction.getType();
}
}  // namespace snn::internal
