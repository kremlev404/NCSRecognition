/*
 * Performed by Anton Kremlev
 */

#pragma once

#include <chrono>
#include <functional>

class Timer {
private:
    int period;

    bool need_to_work;

    void play(const std::function<void()> &callback) const;

public:
    void stop();

    void start(const std::function<void()> &callback);

    explicit Timer(int period);
};