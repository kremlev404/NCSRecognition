/*
 * Performed by Anton Kremlev
 */

#include "gpio.hpp"

#include <wiringPi.h>

GPIO::GPIO() {
    wiringPiSetup();
    pinMode(LedOutput::green_led, OUTPUT);
    digitalWrite(LedOutput::green_led, LOW);
    pinMode(LedOutput::red_led, OUTPUT);
    digitalWrite(LedOutput::red_led, LOW);
}


void GPIO::ledOn(const LedOutput &ledOutput) {
    digitalWrite(ledOutput, HIGH);
}

void GPIO::ledOff(const LedOutput &ledOutput) {
    digitalWrite(ledOutput, LOW);
}
