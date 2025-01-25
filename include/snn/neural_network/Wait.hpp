#pragma once
#include <chrono>

namespace snn
{
enum class waitOperator
{
    noneOp = 0,
    andOp,
    orOp
};

struct Wait
{
        int epochs = -1;
        float accuracy = -1;
        float mae = -1;
        int duration = -1;
        std::chrono::time_point<std::chrono::system_clock> start;
        std::chrono::time_point<std::chrono::system_clock> lastReset;
        std::chrono::time_point<std::chrono::system_clock> lastTick;
        waitOperator op = waitOperator::noneOp;
        auto operator||(const Wait& wait) -> Wait&;
        auto operator&&(const Wait& wait) -> Wait&;
        void startClock();
        [[nodiscard]] auto isOver(int currentEpochs, float CurrentAccuracy, float currentMae) const -> bool;
        [[nodiscard]] auto tick() const -> int;  // Time since last tick in milliseconds
        [[nodiscard]] auto getDuration() -> float;
        [[nodiscard]] auto getDurationAndReset() -> float;
};

extern auto operator""_ep(unsigned long long value) -> Wait;
extern auto operator""_acc(long double value) -> Wait;
extern auto operator""_mae(long double value) -> Wait;
extern auto operator""_ms(unsigned long long value) -> Wait;
extern auto operator""_s(unsigned long long value) -> Wait;
extern auto operator""_min(unsigned long long value) -> Wait;
}  // namespace snn
