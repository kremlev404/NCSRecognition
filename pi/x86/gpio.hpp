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
    void ledOn(LedOutput ledOutput) override {}

    void ledOff(LedOutput ledOutput) override {}
};
