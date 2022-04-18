/*
 * Performed by Anton Kremlev
 */

#pragma once

#include <chrono>
#include <functional>

class Timer {
public:
    void add(std::chrono::milliseconds delay,
             const std::function<void()> &callback);
};