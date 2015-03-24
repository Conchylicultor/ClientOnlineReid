#ifndef FEATURES_H
#define FEATURES_H


#include <iostream>
#include <vector>
#include <map>
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
    // Frame attributes
    array<Mat, 3> histogramChannels;
    array<MajorColorElem, NB_MAJOR_COLORS_EXTRACT> majorColors;

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
    float predictRow(Mat rowFeatureVector) const; // The feature vector will be scaled
    void scaleRow(Mat rowFeatureVector) const;
    void computeDistance(const FeaturesElement &elem1, const FeaturesElement &elem2, Mat &rowFeatureVector) const;
    void extractArray(const float *array,
                      size_t sizeArray,
                      vector<FeaturesElement> &listFeatures) const;

    void setScaleFactors(const Mat &newValue);

    void loadMachineLearning();

private:
    Features();

    int sizeElementArray;

    CvSVM svm;

    Mat scaleFactors;
};

#endif // FEATURES_H
