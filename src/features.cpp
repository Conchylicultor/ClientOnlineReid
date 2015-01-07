#include "features.h"

#include <fstream>

// Variables for features computation
#define HIST_SIZE 100

unsigned int Features::sizeElementArray = 0 // Just to be sure
        + 3*HIST_SIZE // Histogram size
        + NB_MAJOR_COLORS_EXTRACT*3; // Major colors

void Features::computeDistance(const FeaturesElement &elem1, const FeaturesElement &elem2)
{

}

void Features::extractArray(const float *array, const size_t sizeArray, vector<FeaturesElement> &listFeatures)
{
    // No offset

    if(array == nullptr)
    {
        return;
    }

    size_t currentId = 0; // No offset
    for(size_t numberElem = (sizeArray - currentId) / sizeElementArray ; // No offset currently
        numberElem > 0 ;
        --numberElem)
    {
        listFeatures.push_back(FeaturesElement());
        FeaturesElement &currentElem = listFeatures.back();

        // Histogram
        for(size_t channelId = 0 ; channelId < 3 ; ++channelId)
        {
            currentElem.histogramChannels.at(channelId).create(1,100,CV_32FC3); // TODO: Check order x,y !!!!!!!
            for(size_t i = 0 ; i < HIST_SIZE ; ++i)
            {
                currentElem.histogramChannels.at(channelId).at<float>(i) = array[currentId];
                ++currentId;
            }
        }

        // Major colors
        for(size_t i = 0 ; i < NB_MAJOR_COLORS_EXTRACT ; ++i)
        {
            for(size_t j = 0 ; j < 3 ; ++j)// Channel number
            {
                currentElem.majorColors.at(i).color[j] = array[currentId];
                ++currentId;
            }
        }
    }

    cout << listFeatures.size() << endl;
}
