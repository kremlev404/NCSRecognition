/*
 * Performed by Anton Kremlev
 */

#pragma once

#include <memory>

enum LedOutput {
    red_led,
    green_led
};

class IGPIO {
public:
    IGPIO() = default;

    virtual void ledOn(LedOutput ledOutput) = 0;

    virtual void ledOff(LedOutput ledOutput) = 0;
};

std::shared_ptr<IGPIO> build_gpio_controller();