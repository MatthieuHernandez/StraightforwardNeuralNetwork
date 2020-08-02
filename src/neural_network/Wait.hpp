#pragma once

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
        waitOperator op;
        Wait& operator||(const Wait& wait);
        Wait& operator&&(const Wait& wait);
        bool isOver(int epochs, float accuracy, float mae, int duration) const;
    };

    extern Wait operator""_ep(unsigned long long value);
    extern Wait operator""_acc(long double value);
    extern Wait operator""_mae(long double value);
    extern Wait operator""_ms(unsigned long long value);
    extern Wait operator""_s(unsigned long long value);
    extern Wait operator""_min(unsigned long long value);
}
