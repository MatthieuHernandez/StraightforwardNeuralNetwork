#include <memory>
#include "../../src/data/Data.hpp"

class Dataset
{
public:
     std::unique_ptr<snn::Data> data;

     Dataset();
     ~Dataset() = default;

protected:
    virtual void loadData() = 0;
};