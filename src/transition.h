#ifndef TRANSITION_H
#define TRANSITION_H

#include <vector>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

struct CamInfoElement
{
    size_t hashCodeCameraId;

    int beginDate;
    cv::Vec2f entranceVectorOrigin;
    cv::Vec2f entranceVectorEnd;

    int endDate;
    cv::Vec2f exitVectorOrigin;
    cv::Vec2f exitVectorEnd;
};

class Transition
{
public:
    static Transition &getInstance();

    float predict(const CamInfoElement &elem1, const CamInfoElement &elem2) const; // Return a confidence factor to say if a transition can exist between those two elements
    void extractArray(const float *array,
                      size_t &offset, // Modify the offset value
                      CamInfoElement &cameraInfo) const;

    void checkCamera(const CamInfoElement &elem); // Add a camera eventually to the list
    void saveCameraMap() const;
    void clearCameraMap();
    std::map<int, size_t> getCameraMap() const;

private:
    Transition();

    void loadCameraMap(); // TODO: When this function is called (same time as load machine learning)

    std::map<int, size_t> cameraMap;
};

#endif // TRANSITION_H
