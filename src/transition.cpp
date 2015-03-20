#include "transition.h"

size_t reconstructHashcode(const float *array); // TODO: Better inclusion

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

std::map<int, size_t> Transition::getCameraMap() const
{
    return cameraMap;
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
