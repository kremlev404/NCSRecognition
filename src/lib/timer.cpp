/*
 * Performed by Anton Kremlev
 */

#include <thread>
#include <iostream>

#include "timer.hpp"

Timer::Timer(int period) :
        period(period),
        need_to_work(false) {}

void Timer::stop() {
    need_to_work = false;
}

void Timer::start(const std::function<void()> &callback) {
    need_to_work = true;
    play(callback);
}

void Timer::play(const std::function<void()> &callback) const {
    std::thread([this, callback]() {
        Repeat:
        {
            std::thread([=]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(period));
                callback();
            }).join();
            if (need_to_work) {
                std::cout << "Timer still playing..." << std::endl;
                goto Repeat;
            }
        }
    }).detach();
}
