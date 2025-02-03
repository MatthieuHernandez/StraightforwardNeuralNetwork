#include "Wait.hpp"

#include <algorithm>
#include <stdexcept>

namespace snn
{
auto Wait::operator||(const Wait& wait) -> Wait&
{
    if (op == waitOperator::andOp) throw std::runtime_error("Cannot mix || and && operator for waitFor.");

    this->op = waitOperator::orOp;

    if (this->epochs > 0 && wait.epochs > 0)
    {
        this->epochs = std::min(this->epochs, wait.epochs);
    }
    else
    {
        this->epochs = std::max(this->epochs, wait.epochs);
    }

    if (this->accuracy > 0 && wait.accuracy > 0)
    {
        this->accuracy = std::min(this->accuracy, wait.accuracy);
    }
    else
    {
        this->accuracy = std::max(this->accuracy, wait.accuracy);
    }

    this->mae = std::max(this->mae, wait.mae);

    if (this->duration > 0 && wait.duration > 0)
    {
        this->duration = std::min(this->duration, wait.duration);
    }
    else
    {
        this->duration = std::max(this->duration, wait.duration);
    }

    return *this;
}

auto Wait::operator&&(const Wait& wait) -> Wait&
{
    if (op == waitOperator::orOp) throw std::runtime_error("Cannot mix || and && operator for waitFor.");

    this->op = waitOperator::andOp;

    this->epochs = std::max(this->epochs, wait.epochs);
    this->accuracy = std::max(this->accuracy, wait.accuracy);

    if (this->mae > 0 && wait.duration > 0)
    {
        this->mae = std::min(this->mae, wait.mae);
    }
    else
    {
        this->mae = std::max(this->mae, wait.mae);
    }

    this->mae = std::min(this->mae, wait.mae);
    this->duration = std::max(this->duration, wait.duration);

    return *this;
}

void Wait::startClock()
{
    this->start = std::chrono::system_clock::now();
    this->lastTick = this->start;
    this->lastReset = this->start;
}

auto Wait::isOver(int currentEpochs, float CurrentAccuracy, float currentMae) const -> bool
{
    const auto currentDuration = static_cast<int>(
        duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - this->start).count());

    const bool isValidEpochs = currentEpochs > this->epochs && currentEpochs > 0;
    const bool isValidAccuracy = CurrentAccuracy >= this->accuracy && CurrentAccuracy > 0;
    const bool isValidMae = currentMae <= this->mae && currentMae > 0;
    const bool isValidDuration = currentDuration >= this->duration;

    if (this->op == waitOperator::andOp)
    {
        if ((isValidEpochs || this->epochs < 0) && (isValidAccuracy || this->accuracy < 0) &&
            (isValidMae || this->mae < 0) && (isValidDuration || this->duration < 0))
        {
            return true;
        }
    }
    else if ((isValidEpochs && this->epochs >= 0) || (isValidAccuracy && this->accuracy >= 0) ||
             (isValidMae && this->mae >= 0) || (isValidDuration && this->duration >= 0))
    {
        return true;
    }
    return false;
}

auto Wait::tick() const -> int
{
    const auto now = std::chrono::system_clock::now();
    const auto tickDuration = static_cast<int>(duration_cast<std::chrono::milliseconds>(now - this->lastTick).count());
    return tickDuration;
}

auto Wait::getDuration() -> float
{
    const auto now = std::chrono::system_clock::now();
    const auto currentDuration =
        static_cast<float>(duration_cast<std::chrono::milliseconds>(now - this->lastReset).count());
    this->lastTick = now;
    return currentDuration / 1000.0F;  // NOLINT(*magic-numbers)
}

auto Wait::getDurationAndReset() -> float
{
    const auto now = std::chrono::system_clock::now();
    const auto currentDuration =
        static_cast<float>(duration_cast<std::chrono::milliseconds>(now - this->lastReset).count());
    this->lastTick = now;
    this->lastReset = now;
    return currentDuration / 1000.0F;  // NOLINT(*magic-numbers)
}
}  // namespace snn

auto operator""_ep(unsigned long long value) -> snn::Wait
{
    snn::Wait res;
    res.epochs = static_cast<int>(value);
    return res;
}

auto operator""_acc(long double value) -> snn::Wait
{
    snn::Wait res;
    res.accuracy = static_cast<float>(value);
    return res;
}

auto operator""_mae(long double value) -> snn::Wait
{
    snn::Wait res;
    res.mae = static_cast<float>(value);
    return res;
}

auto operator""_ms(unsigned long long value) -> snn::Wait
{
    snn::Wait res;
    res.duration = static_cast<int>(value);
    return res;
}

auto operator""_s(unsigned long long value) -> snn::Wait
{
    snn::Wait res;
    res.duration = static_cast<int>(value) * 1000;  // NOLINT(*magic-numbers)
    return res;
}

auto operator""_min(unsigned long long value) -> snn::Wait
{
    snn::Wait res;
    res.duration = static_cast<int>(value) * 1000 * 60;  // NOLINT(*magic-numbers)
    return res;
}
