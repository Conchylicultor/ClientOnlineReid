#ifndef FEATURES_H
#define FEATURES_H


#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

// Independent features

// Keep has to be lower or equal to extract
#define NB_MAJOR_COLORS_EXTRACT 7
#define NB_MAJOR_COLORS_KEEP 5

struct MajorColorElem
{
    Vec3b color;
    float position;
    int weightColor; // Nb of element of this color
};

// Global structs

struct FeaturesElement
{
    array<Mat, 3> histogramChannels;
    array<MajorColorElem, NB_MAJOR_COLORS_EXTRACT> majorColors;
};

class Features
{
public:
    static void computeDistance(const FeaturesElement &elem1, const FeaturesElement &elem2);
    static void extractArray(const float* &array,
                             const size_t &sizeArray,
                             vector<FeaturesElement> &listFeatures);
};

#endif // FEATURES_H
