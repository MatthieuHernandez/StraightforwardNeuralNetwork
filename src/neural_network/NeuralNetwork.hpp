#pragma once
#include <memory>
#include <boost/serialization/vector.hpp>
#include "Optimizer.hpp"
#include "layer/Layer.hpp"
#include "layer/LayerModel.hpp"
#include "layer/FullyConnected.hpp"
#include "layer/Convolution1D.hpp"
#include "layer/Convolution2D.hpp"
#include "StatisticAnalysis.hpp"
#include "layer/BaseLayer.hpp"
#include "layer/Recurrence.hpp"
#include "layer/GruLayer.hpp"
#include "layer/LocallyConnected1D.hpp"
#include "layer/LocallyConnected2D.hpp"


namespace snn::internal
{
    class NeuralNetwork : public StatisticAnalysis
    {
    private:
        static bool isTheFirst;
        static void initialize();

        void backpropagationAlgorithm(const std::vector<float>& inputs, const std::vector<float>& desired,
                                      bool temporalReset);
        std::vector<float>& calculateError(const std::vector<float>& outputs, const std::vector<float>& desired) const;

        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    protected:
        int maxOutputIndex = -1;

        std::vector<float> output(const std::vector<float>& inputs, bool temporalReset);

        void evaluateOnceForRegression(const std::vector<float>& inputs,
                                       const std::vector<float>& desired,
                                       float precision,
                                       bool temporalReset);
        void evaluateOnceForMultipleClassification(const std::vector<float>& inputs,
                                                   const std::vector<float>& desired,
                                                   float separator,
                                                   bool temporalReset);
        void evaluateOnceForClassification(const std::vector<float>& inputs, int classNumber, bool temporalReset);

    public:
        NeuralNetwork() = default; // use restricted to Boost library only
        NeuralNetwork(std::vector<LayerModel>& architecture);
        NeuralNetwork(const NeuralNetwork& neuralNetwork);
        ~NeuralNetwork() = default;

        [[nodiscard]] int getNumberOfLayers() const;
        [[nodiscard]] int getNumberOfInputs() const;
        [[nodiscard]] int getNumberOfOutputs() const;
        [[nodiscard]] int getNumberOfNeurons() const;
        [[nodiscard]] int getNumberOfParameters() const;

        StochasticGradientDescent optimizer;

        std::vector<std::unique_ptr<BaseLayer>> layers{};

        [[nodiscard]] int isValid() const;

        void trainOnce(const std::vector<float>& inputs, const std::vector<float>& desired, bool temporalReset = false);

        bool operator==(const NeuralNetwork& neuralNetwork) const;
        bool operator!=(const NeuralNetwork& neuralNetwork) const;
    };

    template <class Archive>
    void NeuralNetwork::serialize(Archive& ar, const unsigned int)
    {
        boost::serialization::void_cast_register<NeuralNetwork, StatisticAnalysis>();
        ar & boost::serialization::base_object<StatisticAnalysis>(*this);
        ar & this->optimizer.learningRate;
        ar & this->optimizer.momentum;
        ar & this->maxOutputIndex;
        ar.template register_type<FullyConnected>();
        ar.template register_type<Recurrence>();
        ar.template register_type<GruLayer>();
        ar.template register_type<Convolution1D>();
        ar.template register_type<Convolution2D>();
        ar.template register_type<LocallyConnected1D>();
        ar.template register_type<LocallyConnected2D>();
        ar & layers;
    }
}
