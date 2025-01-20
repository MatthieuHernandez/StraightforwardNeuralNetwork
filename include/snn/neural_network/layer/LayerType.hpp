#pragma once
namespace snn
{
enum layerType
{
    input = 0,
    fullyConnected = 1,
    recurrence = 2,
    gruLayer = 3,
    maxPooling = 4,
    locallyConnected = 5,
    convolution = 6
};
}