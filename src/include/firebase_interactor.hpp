/*
 * Performed by Anton Kremlev
 */

#pragma once

#include <string>
#include <map>

#include "data/person_period_data.hpp"

class FirebaseInteractor {
private:
    static constexpr long long timestamp_corrector_value = 1000000000000;

    std::map<std::string, PersonPeriodData> period_data;

    int max_period;

    static long long get_now();

    void remove_old(const std::string &id);

    float get_avg(const std::string &id);

public:
    explicit FirebaseInteractor(const int &period);

    void send_to_firebase();

    void push(const std::string &id, const float &prob);
};
