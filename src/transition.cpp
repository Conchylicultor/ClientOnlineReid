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
    loadTransitions();
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
    // Clear the folder before write the new transitions
    system("exec rm ../../Data/Debug/Transitions/*");

    vector<Mat> camImgs(cameraMap.size()); // Only one transition
    vector<Mat> finalImgs(cameraMap.size()); // All transitions

    // Loading background image
    for(pair<int, size_t> currentCam : cameraMap) // For each camera
    {
        finalImgs.at(currentCam.first) = backgroundImgs.at(currentCam.first).clone();
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
            camImgs.at(currentCam.first) = backgroundImgs.at(currentCam.first).clone();

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
                imwrite("../../Data/Debug/Transitions/trans_" + to_string(idTransition) + "_" + to_string(currentCam.first) + ".png", camImgs.at(currentCam.first));
            }
        }

        idTransition++;
    }

    cout << "Received cam:" << endl;
    for(pair<int, size_t> currentCam : cameraMap) // For each camera
    {
        cout << currentCam.second << endl;
        imshow("Transition: " + std::to_string(currentCam.second), finalImgs.at(currentCam.first));
    }
}

Mat Transition::plotTransition(const TransitionElement &elem) const
{
    int camInId = -1;
    int camOutId = -1;

    Size finalSize(0,0);
    int offsetInX = 0;

    for(pair<int, size_t> currentCam : cameraMap) // For each camera
    {
        // Has an exit
        if(elem.hashCodeCameraIdOut == currentCam.second)
        {
            camOutId = currentCam.first;
            finalSize.width += backgroundImgs.at(camOutId).cols;
            finalSize.height = std::max(backgroundImgs.at(camOutId).rows, finalSize.height);
        }

        // Has an entrance
        if(elem.hashCodeCameraIdIn == currentCam.second)
        {
            camInId = currentCam.first;
            finalSize.width += backgroundImgs.at(camInId).cols;
            finalSize.height = std::max(backgroundImgs.at(camInId).rows, finalSize.height);
            offsetInX = backgroundImgs.at(camInId).cols;
        }
    }

    Mat finalImgs(finalSize, CV_8UC3);

    Scalar colorExit(0,0,255);
    Scalar colorEntrance(255,0,0);

    if(camInId != -1)
    {
        // Copy the background
        backgroundImgs.at(camOutId).copyTo(finalImgs(Rect(Point(0,0)        , backgroundImgs.at(camOutId).size())));

        Point pt1(elem.exitVectorOrigin[0], elem.exitVectorOrigin[1]);
        Point pt2(elem.exitVectorEnd[0],    elem.exitVectorEnd[1]);

        cv::line(finalImgs, pt1, pt2, colorExit, 2);
        cv::circle(finalImgs, pt2, 5, colorExit);

    }
    if(camOutId != -1)
    {
        // Copy the background
        backgroundImgs.at(camInId) .copyTo(finalImgs(Rect(Point(offsetInX,0), backgroundImgs.at(camInId).size())));

        Point pt1(elem.entranceVectorOrigin[0] + offsetInX, elem.entranceVectorOrigin[1]);
        Point pt2(elem.entranceVectorEnd[0]    + offsetInX, elem.entranceVectorEnd[1]);

        cv::line(finalImgs, pt1, pt2, colorEntrance, 2);
        cv::circle(finalImgs, pt2, 5, colorEntrance);
    }

    return finalImgs;
}

float Transition::predict(const CamInfoElement &elem1, const CamInfoElement &elem2) const
{
    float confidenceCoeff = 0.0;

    // First case: disapearance/reapearance at the same place
    if(elem1.hashCodeCameraId == elem2.hashCodeCameraId)
    {
        float distance = cv::norm(elem1.entranceVectorOrigin - elem2.exitVectorEnd);
        float duration = std::abs(elem1.beginDate - elem2.endDate);
        cout << "Distance: " << distance << endl;
        cout << "Time: " << duration << endl;
        confidenceCoeff = std::exp(-(distance*1.0 + duration*1.0)); // TODO: Weigth and function
    }
    // General case: two different transitions
    else
    {
        TransitionElement newTransition;
        newTransition.hashCodeCameraIdOut = elem2.hashCodeCameraId; // Exit from the last recorded position of the person
        newTransition.exitVectorOrigin    = elem2.exitVectorOrigin;
        newTransition.exitVectorEnd       = elem2.exitVectorEnd;
        newTransition.hashCodeCameraIdIn   = elem1.hashCodeCameraId; // Entrance from the new sequence
        newTransition.entranceVectorOrigin = elem1.entranceVectorOrigin;
        newTransition.entranceVectorEnd    = elem1.entranceVectorEnd;
        newTransition.transitionDuration = elem1.beginDate - elem2.beginDate;

        // Find the closest transition
        float minDistance = -1.0;
        for(TransitionElement currentTransition : listTransitions)
        {
            if(newTransition.hashCodeCameraIdIn == currentTransition.hashCodeCameraIdIn &&
               newTransition.hashCodeCameraIdOut == currentTransition.hashCodeCameraIdOut) // Similar transition (TODO: allow also the reverse transition)
            {
                // Compute the distance between the two transitions
                float currentDistance = 0.0;
                // TODO: Weigth
                currentDistance += cv::norm(currentTransition.exitVectorEnd        - newTransition.exitVectorEnd)        * 1.0;
                currentDistance += cv::norm(currentTransition.entranceVectorOrigin - newTransition.entranceVectorOrigin) * 1.0; // Closest position possible
                currentDistance += std::abs(currentTransition.transitionDuration   - newTransition.transitionDuration)   * 0.1; // Closest duration possible (less important than proximity)

                if(minDistance < 0.0 || // First match
                   currentDistance < minDistance) // Otherwise
                {
                    minDistance = currentDistance;
                }
            }
        }

        // Found: return the distance between the two
        // Not found: Use of a default value
        if(minDistance < 0.0)
        {
            cout << "No closest transition" << endl;
            confidenceCoeff = 0.0; // TODO: Weigth and function
        }
        else
        {
            cout << "Minimal computed distance: " << minDistance << endl;
            confidenceCoeff = std::exp(-minDistance); // TODO: Weight and better fuction

            // Plot the results transitions
            static int compteurResultRecord = 0;

            Mat imageNewTransition = plotTransition(newTransition);
            //Mat imageMinTransition = plotTransition(*minTransition);

            putText(imageNewTransition, "Distance: " + to_string(minDistance), Point(10,20), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 0, 255));
            // The duration is plot into plotTransition

            imwrite("../../Data/Debug/Result_transitions/" + to_string(compteurResultRecord) + ".png", imageNewTransition);
            //imwrite("../../Data/Debug/Result_transitions/" + to_string(compteurResultRecord) + "_min.png", imageMinTransition);

            compteurResultRecord++;
        }
    }

    cout << "Transition probablity: " << confidenceCoeff << endl;
    return confidenceCoeff;
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
    // Load the cameras from file

    FileStorage fileTraining("../../Data/Training/cameras.yml", FileStorage::READ);
    if(!fileTraining.isOpened())
    {
        cout << "Error: Cannot open the cameras on the training file (folder does not exist ?)" << endl;
        return;
    }

    // Load the cameraMap
    cameraMap.clear();
    FileNode nodeCameraMap = fileTraining["cameraMap"];
    for(string currentCam : nodeCameraMap)
    {
        cameraMap.insert(pair<int, size_t>(cameraMap.size(), stoull(currentCam)));
    }

    fileTraining.release();

    // Load the associate background

    system("exec rm ../../Data/Debug/Result_transitions/*");

    backgroundImgs.clear(); // Remove previous background
    backgroundImgs.resize(cameraMap.size());

    for(pair<int, size_t> currentCam : cameraMap) // For each camera
    {
        Mat tempBackground = imread("../../Data/Models/background_" + std::to_string(currentCam.second) + ".png", CV_LOAD_IMAGE_GRAYSCALE); // No color for better vision
        if(!tempBackground.data)
        {
            cout << "Warning: no background image for the cam: " << currentCam.second << ", loading default background..." << endl;
            backgroundImgs.at(currentCam.first) = Mat::zeros(Size(640,480),CV_8UC3);
        }
        else
        {
            backgroundImgs.at(currentCam.first).create(tempBackground.size(), CV_8UC3);
            cvtColor(tempBackground, backgroundImgs.at(currentCam.first), CV_GRAY2RGB); // Now we can plot colors
        }
    }
}

void Transition::loadTransitions()
{
    FileStorage fileTraining("../../Data/Training/calibration.yml", FileStorage::READ);
    if(!fileTraining.isOpened())
    {
        cout << "Error: Cannot open the cameras on the training file (folder does not exist ?)" << endl;
        return;
    }

    // Load the recorded transitions
    listTransitions.clear();

    FileNode nodeTransitions = fileTraining["transitions"];
    for(FileNodeIterator iter = nodeTransitions.begin() ; iter != nodeTransitions.end() ; iter++)
    {
        TransitionElement newTransition;
        newTransition.hashCodeCameraIdOut = stoull((*iter)["camOut"]);
        (*iter)["VecOutX1"] >> newTransition.exitVectorOrigin[0];
        (*iter)["VecOutY1"] >> newTransition.exitVectorOrigin[1];
        (*iter)["VecOutX2"] >> newTransition.exitVectorEnd[0];
        (*iter)["VecOutY2"] >> newTransition.exitVectorEnd[1];
        newTransition.hashCodeCameraIdIn = stoull((*iter)["camIn"]);
        (*iter)["VecInX1"] >> newTransition.entranceVectorOrigin[0];
        (*iter)["VecInY1"] >> newTransition.entranceVectorOrigin[1];
        (*iter)["VecInX2"] >> newTransition.entranceVectorEnd[0];
        (*iter)["VecInY2"] >> newTransition.entranceVectorEnd[1];
        (*iter)["dur"] >> newTransition.transitionDuration;
        listTransitions.push_back(newTransition);
    }

    fileTraining.release();
}
