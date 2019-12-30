#include <memory>
#include "LayerModel.hpp"
#include "Layer.hpp"

namespace snn
{
    extern internal::LayerModel AllToAll(int numberOfNeurons);

    extern internal::LayerModel Recurrent(int numberOfNeurons,  int numberOfRecurrences);

    extern internal::LayerModel Convolution2D(int numberOfConvolution, int sizeOfConvolutionMatrix, int sizeOfInputs[3]);
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