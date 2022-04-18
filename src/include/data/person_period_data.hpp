/*
 * Performed by Anton Kremlev
 */

#pragma once

#include <vector>
#include <list>

struct PersonPeriodData {
    std::list<long> timestamp;
    std::vector<float> avg_prob;
    bool need_to_be_updated = false;

    PersonPeriodData(const long ts, const float prob, bool need_to_be_updated) :
            timestamp({ts}),
            avg_prob({prob}),
            need_to_be_updated(need_to_be_updated) {}

    PersonPeriodData(const std::list<long> &ts, const std::vector<float> &prob, bool need_to_be_updated) :
            timestamp({ts}),
            avg_prob({prob}),
            need_to_be_updated(need_to_be_updated) {}
};