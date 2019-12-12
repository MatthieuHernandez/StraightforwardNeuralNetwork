#include <memory>

enum set
{
    testing = 0,
    training = 1
};

class Dataset
{
public:
     std::unique_ptr<Data> data;

     Dataset();
     ~Dataset() = default;

protected:
    virtual loadData() = 0;
};