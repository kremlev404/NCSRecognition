/*
 * Performed by Anton Kremlev
 */

#pragma once

#include "igpio.hpp"

class GPIO : public IGPIO {
    GPIO() ;
    void ledOn(LedOutput ledOutput) override;

    void ledOff(LedOutput ledOutput) override;
};
