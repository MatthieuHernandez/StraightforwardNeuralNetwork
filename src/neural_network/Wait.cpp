#include <algorithm>
#include <stdexcept>
#include "Wait.hpp"

using namespace snn;
using namespace std;


Wait& Wait::operator||(const Wait& wait)
{
    if (op == andOp)
        throw runtime_error("Cannot mix || and && operator for waitFor.");

    this->op = orOp;

    if (this->epochs > 0 && wait.epochs > 0)
        this->epochs = min(this->epochs, wait.epochs);
    else
        this->epochs = max(this->epochs, wait.epochs);

    if (this->accuracy > 0 && wait.accuracy > 0)
        this->accuracy = min(this->accuracy, wait.accuracy);
    else
        this->accuracy = max(this->accuracy, wait.accuracy);

    if (this->duration > 0 && wait.duration > 0)
        this->duration = min(this->duration, wait.duration);
    else
        this->duration = max(this->duration, wait.duration);

    return *this;
}

Wait& Wait::operator&&(const Wait& wait)
{
    if (op == orOp)
        throw runtime_error("");

    this->op = andOp;

    this->epochs = max(this->epochs, wait.epochs);
    this->accuracy = max(this->accuracy, wait.accuracy);
    this->duration = max(this->duration, wait.duration);

    return *this;
}

bool Wait::isOver(int epochs, float accuracy, float mae, int duration)
{
    const bool isValidEpochs = epochs >= this->epochs;
    const bool isValidAccuracy = accuracy >= this->accuracy;
    const bool isValidMae = mae >= this->mae;
    const bool isValidDuration = duration >= this->duration;

    if (this->op == andOp)
    {
        if ((isValidEpochs || this->epochs < 0)
            && (isValidAccuracy || this->accuracy < 0)
            && (isValidMae || this->mae < 0)
            && (isValidDuration || this->duration < 0))
            return true;
    }
    else if ((isValidEpochs && this->epochs >= 0)
        || (isValidAccuracy && this->accuracy >= 0)
        || (isValidMae && this->mae >= 0)
        || (isValidDuration && this->duration >= 0))
        return true;
    return false;
}

Wait snn::operator""_ep(unsigned long long value)
{
    Wait res;
    res.epochs = value;
    return res;
}

Wait snn::operator""_acc(long double value)
{
    Wait res;
    res.accuracy = value;
    return res;
}

Wait snn::operator""_mae(long double value)
{
    Wait res;
    res.mae = value;
    return res;
}

Wait snn::operator""_ms(unsigned long long value)
{
    Wait res;
    res.duration = value;
    return res;
}

Wait snn::operator""_s(unsigned long long value)
{
    Wait res;
    res.duration = value * 1000;
    return res;
}

Wait snn::operator""_min(unsigned long long value)
{
    Wait res;
    res.accuracy = value * 1000 * 60;
    return res;
}
