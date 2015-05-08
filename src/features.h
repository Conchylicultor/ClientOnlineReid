#ifndef FEATURES_H
#define FEATURES_H


#include <iostream>
#include <vector>
#include <array>
#include <map>
#include "opencv2/opencv.hpp"

// Independent features

// Keep has to be lower or equal to extract
#define NB_MAJOR_COLORS_EXTRACT 7
#define NB_MAJOR_COLORS_KEEP 5

struct MajorColorElem
{
    cv::Vec3b color;
    float position;
    int weightColor; // Nb of element of this color
};

// Global structs

struct FeaturesElement
{
    // Frame attributes
    std::array<cv::Mat, 3> histogramChannels;
    std::array<MajorColorElem, NB_MAJOR_COLORS_EXTRACT> majorColors;

    // Image id information
    int clientId;
    int silhouetteId;
    int imageId;
};

size_t reconstructHashcode(const float *array); // Reconstruct the received hashcode from 2 float received in the array

class Features
{
public:
    static Features &getInstance();

    float predict(const FeaturesElement &elem1, const FeaturesElement &elem2) const;
    float predictRow(cv::Mat rowFeatureVector) const; // The feature vector will be scaled
    void scaleRow(cv::Mat rowFeatureVector) const;
    void computeDistance(const FeaturesElement &elem1, const FeaturesElement &elem2, cv::Mat &rowFeatureVector) const;
    void extractArray(const float *array,
                      size_t sizeArray,
                      std::vector<FeaturesElement> &listFeatures) const;

    void setScaleFactors(const cv::Mat &newValue);

    void loadMachineLearning();

private:
    Features();

    int sizeElementArray;

    CvSVM svm;

    cv::Mat scaleFactors;
};

#endif // FEATURES_H
