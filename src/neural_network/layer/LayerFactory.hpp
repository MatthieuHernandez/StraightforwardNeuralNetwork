#include <memory>
#include "LayerModel.hpp"
#include "Layer.hpp"

namespace snn
{
    extern internal::LayerModel AllToAll(int numberOfNeurons, activationFunction activation = sigmoid);

    extern internal::LayerModel Recurrent(int numberOfNeurons, int numberOfRecurrences, activationFunction activation = sigmoid);

    extern internal::LayerModel Convolution2D(int numberOfConvolution, int sizeOfConvolutionMatrix, int sizeOfInputs[3], activationFunction activation = reLU);
}

namespace snn::internal
{
    class LayerFactory 
    {
    private :
        static std::unique_ptr<Layer> build(LayerModel model);
        static std::unique_ptr<Layer> copy(std::unique_ptr<Layer> layer);

    public :
        static std::vector<std::unique_ptr<Layer>> build(std::vector<LayerModel>& models);
        static std::vector<std::unique_ptr<Layer>> copy(std::vector<std::unique_ptr<Layer>> layers);
    };
}