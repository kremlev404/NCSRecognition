/*
 * Performed by Anton Kremlev
 */

#pragma once

#include "igpio.hpp"

class GPIO : public IGPIO {
public:
    GPIO();

    void ledOn(const LedOutput &ledOutput) override;

    void ledOff(const LedOutput &ledOutput) override;
};
