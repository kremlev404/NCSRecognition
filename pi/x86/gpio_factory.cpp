/*
 * Performed by Anton Kremlev
 */

#include <memory>
#include "igpio.hpp"
#include "gpio.hpp"

std::shared_ptr<IGPIO> build_gpio_controller() {
    return std::make_shared<GPIO>();
}