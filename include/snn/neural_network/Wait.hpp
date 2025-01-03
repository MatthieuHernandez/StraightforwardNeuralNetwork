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
        bool hasStarted = false;
        int epochs = -1;
        float accuracy = -1;
        float mae = -1;
        int duration = -1;
        std::chrono::time_point<std::chrono::system_clock> start;
        std::chrono::time_point<std::chrono::system_clock> lastReset;
        std::chrono::time_point<std::chrono::system_clock> lastTick;
        waitOperator op = waitOperator::noneOp;
        Wait& operator||(const Wait& wait);
        Wait& operator&&(const Wait& wait);
        void startClock();
        [[nodiscard]] bool isOver(int currentEpochs, float CurrentAccuracy, float currentMae) const;
        [[nodiscard]] int tick() const; // Time since last tick in milliseconds
        [[nodiscard]] float getDuration();
        [[nodiscard]] float getDurationAndReset();
    };

    extern Wait operator""_ep(unsigned long long value);
    extern Wait operator""_acc(long double value);
    extern Wait operator""_mae(long double value);
    extern Wait operator""_ms(unsigned long long value);
    extern Wait operator""_s(unsigned long long value);
    extern Wait operator""_min(unsigned long long value);
}
