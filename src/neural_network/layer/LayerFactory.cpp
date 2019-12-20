#include "../../tools/ExtendedExpection.hpp"

static std::unique_ptr<Layer> build(LayerModel)
{
    switch (LayerModel.type)
    {
    case allToAll:
        return
        break;
    
    default:
        NotImplementedException();
    }
}

static std::vector<std::unique_ptr<Layer>> build(LayerModel)
{

}