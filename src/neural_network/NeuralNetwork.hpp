#pragma once
#include <memory>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include "optimizer/NeuralNetworkOptimizer.hpp"
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
#include "layer/MaxPooling1D.hpp"
#include "layer/MaxPooling2D.hpp"
#include "optimizer/NeuralNetworkOptimizerModel.hpp"

namespace snn::internal
{
    class NeuralNetwork : public StatisticAnalysis
    {
    private:
        static bool isTheFirst; // TODO: remplace by seed
        static void initialize();

        bool outputNan = false;

        void backpropagationAlgorithm(const std::vector<float>& inputs, const std::vector<float>& desired, bool temporalReset);
        std::vector<float> calculateError(const std::vector<float>& outputs, const std::vector<float>& desired) const;


        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, unsigned version);

    protected:
        std::vector<float> output(const std::vector<float>& inputs, bool temporalReset);
        std::vector<float> outputForTraining(const std::vector<float>& inputs, bool temporalReset); // Because Dropout is different during training and inference

        void evaluateOnceForRegression(const std::vector<float>& inputs,
                                       const std::vector<float>& desired,
                                       float precision,
                                       bool temporalReset);
        void evaluateOnceForMultipleClassification(const std::vector<float>& inputs,
                                                   const std::vector<float>& desired,
                                                   float separator,
                                                   bool temporalReset);
        void evaluateOnceForClassification(const std::vector<float>& inputs,
                                           int classNumber,
                                           const float separator,
                                           bool temporalReset);

    public:
        NeuralNetwork() = default; // use restricted to Boost library only
        NeuralNetwork(std::vector<LayerModel>& architecture, NeuralNetworkOptimizerModel optimizer);
        NeuralNetwork(const NeuralNetwork& neuralNetwork);
        virtual ~NeuralNetwork() = default;
        
        [[nodiscard]] bool hasNan() const;
        [[nodiscard]] int getNumberOfLayers() const;
        [[nodiscard]] int getNumberOfInputs() const;
        [[nodiscard]] int getNumberOfOutputs() const;
        [[nodiscard]] int getNumberOfNeurons() const;
        [[nodiscard]] int getNumberOfParameters() const;

        std::shared_ptr<NeuralNetworkOptimizer> optimizer = nullptr;

        std::vector<std::unique_ptr<BaseLayer>> layers{};

        [[nodiscard]] int isValid() const;

        void trainOnce(const std::vector<float>& inputs, const std::vector<float>& desired, bool temporalReset = true);

        bool operator==(const NeuralNetwork& neuralNetwork) const;
        bool operator!=(const NeuralNetwork& neuralNetwork) const;
    };

    template <class Archive>
    void NeuralNetwork::serialize(Archive& ar, [[maybe_unused]] const unsigned version)
    {
        if (isTheFirst)
            this->initialize();
        boost::serialization::void_cast_register<NeuralNetwork, StatisticAnalysis>();
        ar & boost::serialization::base_object<StatisticAnalysis>(*this);
        ar.template register_type<FullyConnected>();
        ar.template register_type<Recurrence>();
        ar.template register_type<GruLayer>();
        ar.template register_type<Convolution1D>();
        ar.template register_type<Convolution2D>();
        ar.template register_type<LocallyConnected1D>();
        ar.template register_type<LocallyConnected2D>();
        ar.template register_type<StochasticGradientDescent>();
        ar.template register_type<MaxPooling1D>();
        ar.template register_type<MaxPooling2D>();
        ar & layers;
        ar & this->optimizer;
    }
}
