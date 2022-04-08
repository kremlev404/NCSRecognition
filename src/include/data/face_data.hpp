#pragma once

#include <vector>
#include <string>

struct FaceData {
    std::string file_name;
    std::vector<float> descriptor;
};