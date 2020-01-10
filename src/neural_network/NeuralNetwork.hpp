#pragma once
#include <memory>
#include <boost/serialization/vector.hpp>
#include "Optimizer.hpp"
#include "layer/Layer.hpp"
#include "layer/LayerModel.hpp"
#include "layer/AllToAll.hpp"
#include "StatisticAnalysis.hpp"


namespace snn::internal
{
    class NeuralNetwork : public StatisticAnalysis
    {
    private :
        static bool isTheFirst;
        static void initialize();

        void backpropagationAlgorithm(const std::vector<float>& inputs, const std::vector<float>& desired);
        std::vector<float>& calculateError(const std::vector<float>& outputs, const std::vector<float>& desired) const;

        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    protected :
        int maxOutputIndex;

        [[nodiscard]] std::vector<float> output(const std::vector<float>& inputs);

        void evaluateOnceForRegression(const std::vector<float>& inputs,
                                       const std::vector<float>& desired,
                                       float precision);
        void evaluateOnceForMultipleClassification(const std::vector<float>& inputs,
                                                   const std::vector<float>& desired,
                                                   float separator);
        void evaluateOnceForClassification(const std::vector<float>& inputs, int classNumber);

        int getLastError() const;

    public:
        NeuralNetwork() = default; // use restricted to Boost library only
        NeuralNetwork(int numberOfInputs, std::vector<LayerModel>& models);
        NeuralNetwork(const NeuralNetwork& neuralNetwork);
        ~NeuralNetwork() = default;

        int getNumberOfLayers() const;
        int getNumberOfInputs() const;
        int getNumberOfOutputs() const;

        StochasticGradientDescent optimizer;

        std::vector<std::unique_ptr<Layer>> layers{};
        int isValid() const;

        void trainOnce(const std::vector<float>& inputs, const std::vector<float>& desired);

        bool operator==(const NeuralNetwork& neuralNetwork) const;
        bool operator!=(const NeuralNetwork& neuralNetwork) const;
    };

    template <class Archive>
    void NeuralNetwork::serialize(Archive& ar, const unsigned int version)
    {
        boost::serialization::void_cast_register<NeuralNetwork, StatisticAnalysis>();
        ar & boost::serialization::base_object<StatisticAnalysis>(*this);
        ar & this->optimizer.learningRate;
        ar & this->optimizer.momentum;
        ar & this->maxOutputIndex;
        ar.template register_type<internal::AllToAll>();
        ar & layers;
    }
}
