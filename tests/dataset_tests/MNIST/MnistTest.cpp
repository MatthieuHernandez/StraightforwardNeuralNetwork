#include "../../GTestTools.hpp"
#include "data/Data.hpp"
#include "Mnist.hpp"

using namespace std;
using namespace snn;

class MnistTest : public testing::Test
{
protected :
    MnistTest()
    {
        Mnist dataset;
        data = move(dataset.data);
    }

public :
    unique_ptr<Data> data;
};

TEST_F(MnistTest, loadData)
{
    ASSERT_TRUE(data);
    ASSERT_EQ(data->sizeOfData, 784);
    ASSERT_EQ(data->numberOfLabel, 10);
    ASSERT_EQ(data->sets[training].inputs.size(), 60000);
    ASSERT_EQ(data->sets[training].labels.size(), 60000);
    ASSERT_EQ(data->sets[snn::testing].inputs.size(), 10000);
    ASSERT_EQ(data->sets[snn::testing].labels.size(), 10000);
}