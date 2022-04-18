/*
 * Performed by Anton Kremlev
 */

#include "timer.hpp"

#include <thread>

void Timer::add(std::chrono::milliseconds delay,
                const std::function<void()> &callback) {
    std::thread([=]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(delay));
        callback();
    }).detach();
}