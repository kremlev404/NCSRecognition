/*
 * Performed by Anton Kremlev
 */

#include <memory>
#include "gpio.hpp"
#include "igpio.hpp"

std::shared_ptr<IGPIO> build_gpio_controller() {
    return std::make_shared<GPIO>();
}