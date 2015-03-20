#include "transition.h"

size_t reconstructHashcode(const float *array); // TODO: Better inclusion

static const int transitionDurationMin = -10;
static const int transitionDurationMax = 10; // Duration limits of the transition

Transition &Transition::getInstance()
{
    static Transition instance;
    return instance;
}

Transition::Transition()
{
    loadCameraMap();
}

void Transition::extractArray(const float *array, size_t &offset, CamInfoElement &cameraInfo) const
{
    // Offset values (Global attributes)
    cameraInfo.hashCodeCameraId = reconstructHashcode(&array[0]);

    float castValue;

    castValue = array[2];
    cameraInfo.beginDate = reinterpret_cast<int&>(castValue);

    castValue = array[3];
    cameraInfo.endDate = reinterpret_cast<int&>(castValue);

    cameraInfo.entranceVectorOrigin = Vec2f(array[4], array[5]);
    cameraInfo.entranceVectorEnd    = Vec2f(array[6], array[7]);

    cameraInfo.exitVectorOrigin     = Vec2f(array[8], array[9]);
    cameraInfo.exitVectorEnd        = Vec2f(array[10], array[11]);

    offset += 12; // New offset value
}

void Transition::recordTransitions(const vector<vector<CamInfoElement> > &listSequencePerson)
{
    listTransitions.clear(); // We don't want use previous transitions

    for(vector<CamInfoElement> const &currentPersonCamInfoList : listSequencePerson)
    {
        for(size_t i = 0 ; i < currentPersonCamInfoList.size() ; ++i)
        {
            // TODO: What is the best way to determine the transitions ? Can the algorithm be improved ?
            // Can we have multiple valid transitions for one sequence (not just between two cameras) ?

            // Looking for the smallest transition
            int closestCamInfo = -1;
            int closestCamInfoDuration = -1;
            for(size_t j = 0 ; j < currentPersonCamInfoList.size() ; ++j)
            {
                // Find the shortest transition
                if(i != j && currentPersonCamInfoList.at(i).beginDate < currentPersonCamInfoList.at(j).beginDate)
                {
                    int currentDuration = currentPersonCamInfoList.at(j).beginDate - currentPersonCamInfoList.at(i).beginDate; // > 0 (due to previous condition)

                    // First time
                    if(closestCamInfo == -1)
                    {
                        closestCamInfoDuration = currentDuration;
                        closestCamInfo = j;
                    }
                    else if(currentDuration < closestCamInfoDuration)
                    {
                        closestCamInfoDuration = currentDuration;
                        closestCamInfo = j;
                    }
                }
            }

            // Creation of the new transition
            TransitionElement newTransition;

            // The transition is between an exit and a re-entrance
            const CamInfoElement &camInfoElemtOut = currentPersonCamInfoList.at(i);
            newTransition.hashCodeCameraIdOut = camInfoElemtOut.hashCodeCameraId;
            newTransition.exitVectorOrigin = camInfoElemtOut.exitVectorOrigin;
            newTransition.exitVectorEnd = camInfoElemtOut.exitVectorEnd;

            // Match
            if(closestCamInfo != -1)
            {
                // The transition is between an exit and a re-entrance
                const CamInfoElement &camInfoElemtIn = currentPersonCamInfoList.at(closestCamInfo);
                newTransition.hashCodeCameraIdIn = camInfoElemtIn.hashCodeCameraId;
                newTransition.entranceVectorOrigin = camInfoElemtIn.entranceVectorOrigin;
                newTransition.entranceVectorEnd = camInfoElemtIn.entranceVectorEnd;

                newTransition.transitionDuration = camInfoElemtIn.beginDate - camInfoElemtOut.endDate; // != closestCamInfosDuration

                // Filter the transition if the duration is too long
                if(newTransition.transitionDuration > transitionDurationMax ||
                   newTransition.transitionDuration < transitionDurationMin)
                {
                    cout << "Transition too long(disappearance): " << newTransition.transitionDuration << endl;
                    closestCamInfo = -1; // Add the transition as disappearance transition
                }
            }

            // No match: disappearance
            if(closestCamInfo == -1)
            {
                newTransition.hashCodeCameraIdIn = 0; // < No reappareance
                newTransition.entranceVectorOrigin = cv::Vec2f(0.0, 0.0);
                newTransition.entranceVectorEnd = cv::Vec2f(0.0, 0.0);
                newTransition.transitionDuration = 0;
            }

            listTransitions.push_back(newTransition);
        }
    }

    // Record the transitions: Append to existing file
    FileStorage fileTraining("../../Data/Training/calibration.yml", FileStorage::WRITE);
    if(!fileTraining.isOpened())
    {
        cout << "Error: Cannot record the calibration file (folder does not exist ?)" << endl;
    }

    fileTraining << "transitions" << "[";
    for(TransitionElement const &currentTransition : listTransitions)
    {
        fileTraining << "{:";
        fileTraining << "camOut" << std::to_string(currentTransition.hashCodeCameraIdOut);
        fileTraining << "VecOutX1" << currentTransition.exitVectorOrigin[0];
        fileTraining << "VecOutY1" << currentTransition.exitVectorOrigin[1];
        fileTraining << "VecOutX2" << currentTransition.exitVectorEnd[0];
        fileTraining << "VecOutY2" << currentTransition.exitVectorEnd[1];
        fileTraining << "camIn" << std::to_string(currentTransition.hashCodeCameraIdIn);
        fileTraining << "VecInX1" << currentTransition.entranceVectorOrigin[0];
        fileTraining << "VecInY1" << currentTransition.entranceVectorOrigin[1];
        fileTraining << "VecInX2" << currentTransition.entranceVectorEnd[0];
        fileTraining << "VecInY2" << currentTransition.entranceVectorEnd[1];
        fileTraining << "dur" << currentTransition.transitionDuration;
        fileTraining << "}";
    }
    fileTraining << "]";

    fileTraining.release();
}

void Transition::plotTransitions()
{
    vector<Mat> backgroundImgs(cameraMap.size());
    vector<Mat> camImgs(cameraMap.size()); // Only one transition
    vector<Mat> finalImgs(cameraMap.size()); // All transitions

    // Clear the folder before write the new transitions
    system("exec rm ../../Data/Transitions/*");

    // Loading background image
    for(pair<int, size_t> currentCam : cameraMap) // For each camera
    {
        backgroundImgs.at(currentCam.first) = imread("../../Data/Models/background_" + std::to_string(currentCam.second) + ".png", CV_LOAD_IMAGE_GRAYSCALE); // No color for better vision
        if(!backgroundImgs.at(currentCam.first).data)
        {
            cout << "Error: no background image for the cam: " << currentCam.second << ", loading default background..." << endl;
            backgroundImgs.at(currentCam.first) = Mat::zeros(Size(640,480),CV_8UC1);
        }

        camImgs.  at(currentCam.first).create(backgroundImgs.at(currentCam.first).size(), CV_8UC3);

        finalImgs.at(currentCam.first).create(backgroundImgs.at(currentCam.first).size(), CV_8UC3);

        cv::cvtColor(backgroundImgs.at(currentCam.first), finalImgs.at(currentCam.first), CV_GRAY2RGB); // Now we can plot colors
    }

    int idTransition = 0; // For saving the transition

    for(TransitionElement const &currentTransition : listTransitions)
    {
        // Choose a random color for the arrow
        Scalar color;
        color[0] = std::rand() % 255;
        color[1] = std::rand() % 255;
        color[2] = std::rand() % 255;

        Scalar colorExit(0,0,255);
        Scalar colorEntrance(255,0,0);

        Scalar colorSolitary(255,0,250); // Color if the transition is one way (ex: just disappearance)
        Scalar colorArrow; // Final color (= solitary or random depending of the transition)

        for(pair<int, size_t> currentCam : cameraMap) // For each camera
        {
            // Clear the background
            cv::cvtColor(backgroundImgs.at(currentCam.first), camImgs.at(currentCam.first), CV_GRAY2RGB); // Now we can plot colors

            // Has an exit
            if(currentTransition.hashCodeCameraIdOut == currentCam.second)
            {
                Point pt1(currentTransition.exitVectorOrigin[0], currentTransition.exitVectorOrigin[1]);
                Point pt2(currentTransition.exitVectorEnd[0],    currentTransition.exitVectorEnd[1]);

                // Plot the arrow into the right cam
                if(currentTransition.hashCodeCameraIdIn)
                {
                    colorArrow = color;
                }
                else
                {
                    colorArrow = colorSolitary;
                }
                cv::line(camImgs.at(currentCam.first), pt1, pt2, colorArrow, 2);
                cv::circle(camImgs.at(currentCam.first), pt2, 5, colorEntrance);

                cv::line(finalImgs.at(currentCam.first), pt1, pt2, colorArrow, 2);
                cv::circle(finalImgs.at(currentCam.first), pt2, 5, colorEntrance);

                // Save the image
                imwrite("../../Data/Transitions/trans_" + to_string(idTransition) + "_" + to_string(currentCam.first) + ".png", camImgs.at(currentCam.first));
            }

            // Has an entrance
            if(currentTransition.hashCodeCameraIdIn == currentCam.second)
            {
                Point pt1(currentTransition.entranceVectorOrigin[0], currentTransition.entranceVectorOrigin[1]);
                Point pt2(currentTransition.entranceVectorEnd[0],    currentTransition.entranceVectorEnd[1]);

                // Plot the arrow into the right cam
                if(currentTransition.hashCodeCameraIdOut)
                {
                    colorArrow = color;
                }
                else
                {
                    colorArrow = colorSolitary;
                }
                cv::line(camImgs.at(currentCam.first), pt1, pt2, colorArrow, 2);
                cv::circle(camImgs.at(currentCam.first), pt2, 5, colorExit);

                cv::line(finalImgs.at(currentCam.first), pt1, pt2, colorArrow, 2);
                cv::circle(finalImgs.at(currentCam.first), pt2, 5, colorExit);

                // Save the image
                imwrite("../../Data/Transitions/trans_" + to_string(idTransition) + "_" + to_string(currentCam.first) + ".png", camImgs.at(currentCam.first));
            }
        }

        idTransition++;
    }

    for(pair<int, size_t> currentCam : cameraMap) // For each camera
    {
        imshow("Transition: " + std::to_string(currentCam.second), finalImgs.at(currentCam.first));
    }
}

float Transition::predict(const CamInfoElement &elem1, const CamInfoElement &elem2) const
{
    return 0.0;
}

void Transition::checkCamera(const CamInfoElement &elem)
{
    bool newCam = true;

    // Does the camera already exist ?
    for(pair<int, size_t> currentElem : cameraMap)
    {
        if(elem.hashCodeCameraId == currentElem.second)
        {
            newCam = false;
        }
    }

    // Otherwise, we simply add it
    if(newCam)
    {
        cameraMap.insert(pair<int, size_t>(cameraMap.size(), elem.hashCodeCameraId));
    }
}

void Transition::saveCameraMap() const
{
    FileStorage fileTraining("../../Data/Training/cameras.yml", FileStorage::WRITE);

    // Record the map of the camera
    if(!fileTraining.isOpened())
    {
        cout << "Error: Cannot record the cameras on the training file (folder does not exist ?)" << endl;
        return;
    }

    fileTraining << "cameraMap" << "[";
    for(pair<int, size_t> currentElem : cameraMap) // For each camera
    {
        fileTraining << std::to_string(currentElem.second);
    }
    fileTraining << "]";
}

void Transition::clearCameraMap()
{
    cameraMap.clear();
}

void Transition::loadCameraMap()
{
    FileStorage fileTraining("../../Data/Training/cameras.yml", FileStorage::READ);

    // Load the cameraMap
    cameraMap.clear();
    FileNode nodeCameraMap = fileTraining["cameraMap"];
    for(string currentCam : nodeCameraMap)
    {
        cameraMap.insert(pair<int, size_t>(cameraMap.size(), stoull(currentCam)));
    }

    fileTraining.release();
}
