#include <memory>

namespace snn
{
    extern LayerModel AlltoAll(int NumberOfNeurons);

    extern LayerModel Recurrent(int NumberOfNeurons,  int numberOfRecurences);

    extern LayerModel Convolution2D(int numberOfConvolution, sizeOfConvolutionMatrix, sizeOfInputs[3]);
}

namespace snn::internal
{
    class LayerFactory 
    {
    private :
        static std::unique_ptr<Layer> build(LayerModel);

    public :
        static std::vector<std::unique_ptr<Layer>> build(LayerModel);
    };
}