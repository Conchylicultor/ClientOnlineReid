#ifndef TRANSITION_H
#define TRANSITION_H

#include <vector>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

struct TransitionElement
{
    // Information on the first stage of the transition (leave the camera)
    size_t hashCodeCameraIdOut;
    cv::Vec2f exitVectorOrigin;
    cv::Vec2f exitVectorEnd;

    // Information on the final stage of the transition (reappearance)
    size_t hashCodeCameraIdIn;
    cv::Vec2f entranceVectorOrigin;
    cv::Vec2f entranceVectorEnd;

    int transitionDuration; // Can be negative if the person reappear in a camera before leaving the previous one
};

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

    void recordTransitions(const vector<vector<CamInfoElement> > &listSequencePerson);
    void plotTransitions();

    void checkCamera(const CamInfoElement &elem); // Add a camera eventually to the list
    void saveCameraMap() const;
    void clearCameraMap();

private:
    Transition();

    void loadCameraMap();
    void loadTransitions();

    std::map<int, size_t> cameraMap;
    vector<TransitionElement> listTransitions;
};

#endif // TRANSITION_H
