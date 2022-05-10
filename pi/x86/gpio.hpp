/*
 * Performed by Anton Kremlev
 */

#pragma once

#include "igpio.hpp"

/*
 * It is mock class for non raspberry system
 * witch doesn't have library for GPIO led control
 */

class GPIO : public IGPIO {
    void ledOn(const LedOutput &ledOutput) override {}

    void ledOff(const LedOutput &ledOutput) override {}
};
