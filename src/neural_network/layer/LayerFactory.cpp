#include "LayerFactory.hpp"
#include "../../tools/ExtendedExpection.hpp"

using namespace std;

static unique_ptr<Layer> build(LayerModel)
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

static vector<unique_ptr<Layer>> build(LayerModel)
{

}