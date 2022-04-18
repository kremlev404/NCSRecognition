/*
 * Performed by Anton Kremlev
 */

#include <algorithm>
#include <chrono>
#include <numeric>
#include <iostream>

#include "firebase_interactor.hpp"

FirebaseInteractor::FirebaseInteractor(int period) : max_period(period) {}

long long FirebaseInteractor::get_now() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
}

void FirebaseInteractor::remove_old(const std::string &id) {
    int64_t now = get_now();

    auto newEnd = std::remove_if(period_data.at(id).timestamp.begin(), period_data.at(id).timestamp.end(),
                                 [&](int num) {
                                     return now - max_period >= num;
                                 });

    period_data.at(id).timestamp.erase(newEnd, period_data.at(id).timestamp.end());
}

float FirebaseInteractor::get_avg(const std::string &id) {
    const auto v = period_data.find(id)->second.avg_prob;

    return std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
}

void FirebaseInteractor::send_to_firebase() {
    for (auto &person: period_data) {
        if (person.second.need_to_be_updated) {
            const std::string person_id = person.first;
            remove_old(person_id);

            const auto timestamp_value = get_now();
            std::string timestamp_message;
            if (timestamp_value / timestamp_corrector_value > 0)
                timestamp_message.append(std::to_string(timestamp_value));
            else
                timestamp_message.append(std::to_string(timestamp_value * 1000));

            std::string prob = std::to_string(get_avg(person_id));

            auto call_script = std::string("/usr/bin/python3 ../../py/main.py -id ")
                    .append(person_id)
                    .append(" -p ")
                    .append(prob)
                    .append(" -t ")
                    .append(timestamp_message);

            std::cout << "FirebaseInteractor::send_to_firebase " << call_script << std::endl;
            (void) system(call_script.c_str());
            person.second.need_to_be_updated = false;
        }
    }
}

void FirebaseInteractor::push(const std::string &id, float prob) {
    auto val = period_data.find(id);
    if (val != period_data.end()) {
        remove_old(id);
        period_data.at(val->first).avg_prob.push_back(prob);
        period_data.at(val->first).timestamp.push_back(get_now());
        period_data.at(val->first).need_to_be_updated = true;
    } else {
        period_data.insert(std::pair<std::string, PersonPeriodData>(id, PersonPeriodData(get_now(), prob, true)));
    }
}
