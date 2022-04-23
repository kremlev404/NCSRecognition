/*
 * Performed by Anton Kremlev
 */

#pragma once

#include <memory>

enum LedOutput {
    red_led = 7, // 7 pin, wiringN is 7
    green_led = 3 // 15 pin, wiringN is 3
};

class IGPIO {
public:
    IGPIO() = default;

    virtual void ledOn(LedOutput ledOutput) = 0;

    virtual void ledOff(LedOutput ledOutput) = 0;
};

std::shared_ptr<IGPIO> build_gpio_controller();
