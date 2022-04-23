/*
 * Performed by Anton Kremlev
 */

#include "gpio.hpp"

#include <wiringPi.h>

GPIO::GPIO() {
    wiringPiSetup();
    pinMode(LedOutput::green_led, OUTPUT);
    pinMode(LedOutput::red_led, OUTPUT);
}


void GPIO::ledOn(LedOutput ledOutput) {
    digitalWrite(ledOutput, 0);
}

void GPIO::ledOff(LedOutput ledOutput) {
    digitalWrite(ledOutput, 0);
}