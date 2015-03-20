#include "features.h"

#include <fstream>

// Variables for features computation
const int HIST_SIZE = 100;
const bool CAMERA_FEATURES = false;

size_t reconstructHashcode(const float *array)
{
    float castValue; // Avoid warning errors (cast a const)
    castValue = array[0];
    unsigned int leastSignificantBits = reinterpret_cast<unsigned int&>(castValue);
    castValue = array[1];
    unsigned int mostSignificantBits  = reinterpret_cast<unsigned int&>(castValue);

    size_t reconstructValue = mostSignificantBits;
    reconstructValue <<= 32;
    reconstructValue |= leastSignificantBits;

    return reconstructValue;
}

Features &Features::getInstance()
{
    static Features instance;
    return instance;
}

Features::Features() :
    sizeElementArray(0)
{
    sizeElementArray += 3*HIST_SIZE; // Histogram size
    sizeElementArray += NB_MAJOR_COLORS_EXTRACT*3; // Major colors

    loadMachineLearning();
}

float Features::predict(const FeaturesElement &elem1, const FeaturesElement &elem2) const
{
    Mat rowFeatureVector;

    computeDistance(elem1, elem2, rowFeatureVector);

    return predictRow(rowFeatureVector); // Scale and predict
}

float Features::predictRow(Mat rowFeatureVector) const
{
    // Scale
    scaleRow(rowFeatureVector);
    return svm.predict(rowFeatureVector);
}

void Features::scaleRow(Mat rowFeatureVector) const
{
    // Simple scale by dividing by maxValue
    for(size_t i = 0 ; i < static_cast<unsigned>(rowFeatureVector.cols) ; ++i)
    {
        rowFeatureVector.at<float>(i) /= scaleFactors.at<float>(0,i);
    }
}

void Features::computeDistance(const FeaturesElement &elem1, const FeaturesElement &elem2, Mat &rowFeatureVector) const
{
    int dimentionFeatureVector = 3 // Histogram
                               + NB_MAJOR_COLORS_KEEP; // Major colors

    // Additionnal feature = additionnal dimention
    if(CAMERA_FEATURES)
    {
        // We add the categorical feature. For instance (0,0,1,0) for cam3 or (1,0,0,0) for cam1
        dimentionFeatureVector += cameraMap.size()*2; // Factor 2 is for entrance camera and exit camera
        dimentionFeatureVector += 1; // For the duration between entrance and exit time
        dimentionFeatureVector += 8; // For the entrance and exit vector (2 for each points)
    }

    rowFeatureVector = cv::Mat::zeros(1, dimentionFeatureVector, CV_32FC1);

    int currentIndexFeature = 0;// Usefull if I change the order or remove a feature (don't need to change all the index)

    // Histogram

    rowFeatureVector.at<float>(0, currentIndexFeature+0) = compareHist(elem1.histogramChannels.at(0), elem2.histogramChannels.at(0), CV_COMP_BHATTACHARYYA);
    rowFeatureVector.at<float>(0, currentIndexFeature+1) = compareHist(elem1.histogramChannels.at(1), elem2.histogramChannels.at(1), CV_COMP_BHATTACHARYYA);
    rowFeatureVector.at<float>(0, currentIndexFeature+2) = compareHist(elem1.histogramChannels.at(2), elem2.histogramChannels.at(2), CV_COMP_BHATTACHARYYA);
    currentIndexFeature += 3;

    // Major Colors

    // Compute only with the most weigthed on
    for (size_t i = 0; i < NB_MAJOR_COLORS_KEEP; ++i)
    {
        float minDist = norm(elem1.majorColors.at(i).color - elem2.majorColors.front().color);
        float dist = 0.0;
        for (size_t j = 0; j < NB_MAJOR_COLORS_KEEP; ++j)
        {
            dist = norm(elem1.majorColors.at(i).color - elem2.majorColors.at(j).color);
            if(dist < minDist)
            {
                minDist = dist;
            }
        }
        rowFeatureVector.at<float>(0,currentIndexFeature) = minDist;
        currentIndexFeature++;
    }

    if(CAMERA_FEATURES)
    {
        const FeaturesElement *firstElem = 0;
        const FeaturesElement *lastElem = 0;
        // Check who came first
        // The one who came first is the first seen (by any camera)
        // t:--------------------------------------------->
        // Seq 1,2:        *---------------* *---*
        // Seq 3:                     *------------*
        // Seq 4:                *-----*
        // The order of the sequences above is 1,4,3,2
        if(elem1.beginDate < elem2.beginDate) // Elem1 > Elem2
        {
            firstElem = &elem1;
            lastElem = &elem2;
        }
        else // Elem2 > Elem1
        {
            firstElem = &elem2;
            lastElem = &elem1;
            // If begin at the same time, arbitrary choice
        }

        // Entrance and exit cam

        for(pair<int, size_t> currentElem : cameraMap) // For each camera
        {
            if(currentElem.second == firstElem->hashCodeCameraId) // 1 for the camera
            {
                rowFeatureVector.at<float>(0, currentIndexFeature) = 1.0;
            }
            else // 0 for the other
            {
                rowFeatureVector.at<float>(0, currentIndexFeature) = 0.0;
            }
            currentIndexFeature++;
        }

        for(pair<int, size_t> currentElem : cameraMap) // For each camera
        {
            if(currentElem.second == lastElem->hashCodeCameraId) // 1 for the camera
            {
                rowFeatureVector.at<float>(0, currentIndexFeature) = 1.0;
            }
            else // 0 for the other
            {
                rowFeatureVector.at<float>(0, currentIndexFeature) = 0.0;
            }
            currentIndexFeature++;
        }


        // Entrance and exit time
        rowFeatureVector.at<float>(0, currentIndexFeature) = lastElem->beginDate - firstElem->endDate;
        currentIndexFeature++;

        // Entrance and exit vector
        rowFeatureVector.at<float>(0, currentIndexFeature+0) = firstElem->exitVectorOrigin[0];
        rowFeatureVector.at<float>(0, currentIndexFeature+1) = firstElem->exitVectorOrigin[1];
        rowFeatureVector.at<float>(0, currentIndexFeature+2) = firstElem->exitVectorEnd[0];
        rowFeatureVector.at<float>(0, currentIndexFeature+3) = firstElem->exitVectorEnd[1];
        currentIndexFeature += 4;

        rowFeatureVector.at<float>(0, currentIndexFeature+0) = lastElem->entranceVectorOrigin[0];
        rowFeatureVector.at<float>(0, currentIndexFeature+1) = lastElem->entranceVectorOrigin[1];
        rowFeatureVector.at<float>(0, currentIndexFeature+2) = lastElem->entranceVectorEnd[0];
        rowFeatureVector.at<float>(0, currentIndexFeature+3) = lastElem->entranceVectorEnd[1];
        currentIndexFeature += 4;
    }

    // The feature scaling is not made in this function
}

void Features::extractArray(const float *array, size_t sizeArray, vector<FeaturesElement> &listFeatures) const
{
    // Test the validity
    if(array == nullptr)
    {
        return;
    }

    // Offset values (Global attributes)
    size_t hashCodeCameraId = reconstructHashcode(&array[0]);
    float castValue;
    castValue = array[2];
    int beginDate = reinterpret_cast<int&>(castValue);
    castValue = array[3];
    int endDate = reinterpret_cast<int&>(castValue);
    cv::Vec2f entranceVectorOrigin(array[4], array[5]);
    cv::Vec2f entranceVectorEnd(array[6], array[7]);
    cv::Vec2f exitVectorOrigin(array[8], array[9]);
    cv::Vec2f exitVectorEnd(array[10], array[11]);

    // Shift the origin to remove the offset
    size_t offset = 12;
    array = &array[offset];
    sizeArray -= offset;

    size_t currentId = 0; // No offset anymore
    for(size_t numberElem = (sizeArray - currentId) / sizeElementArray; // Number of frame's feature send
        numberElem > 0 ;
        --numberElem)
    {
        listFeatures.push_back(FeaturesElement());
        FeaturesElement &currentElem = listFeatures.back();

        // Copy the global attributes
        currentElem.hashCodeCameraId = hashCodeCameraId;
        currentElem.beginDate = beginDate;
        currentElem.endDate = endDate;
        currentElem.entranceVectorOrigin = entranceVectorOrigin;
        currentElem.entranceVectorEnd    = entranceVectorEnd;
        currentElem.exitVectorOrigin     = exitVectorOrigin;
        currentElem.exitVectorEnd        = exitVectorEnd;

        // Histogram
        for(size_t channelId = 0 ; channelId < 3 ; ++channelId)
        {
            currentElem.histogramChannels.at(channelId).create(1,100,CV_32F);
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
}

void Features::checkCamera(const FeaturesElement &elem)
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

void Features::saveCameraMap(FileStorage &fileTraining) const
{
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

void Features::clearCameraMap()
{
    cameraMap.clear();
}

std::map<int, size_t> Features::getCameraMap() const
{
    return cameraMap;
}

void Features::setScaleFactors(const Mat &newValue)
{
    scaleFactors = newValue;
}

void Features::loadMachineLearning()
{
    // Loading file
    FileStorage fileTraining("../../Data/Training/training.yml", FileStorage::READ);

    if(!fileTraining.isOpened())
    {
        cout << "Error: cannot open the training file" << endl;
        return; // No exit: we could be in training mode
    }
    cout << "Training classifier..." << endl;

    // Loading the scales factors
    fileTraining["scaleFactors"] >> scaleFactors;

    // Loading the scale training set
    Mat trainingData;
    Mat trainingClasses;
    fileTraining["trainingData"]    >> trainingData;
    fileTraining["trainingClasses"] >> trainingClasses;

    // Load the cameraMap
    cameraMap.clear();
    FileNode nodeCameraMap = fileTraining["cameraMap"];
    for(string currentCam : nodeCameraMap)
    {
        cameraMap.insert(pair<int, size_t>(cameraMap.size(), stoull(currentCam)));
    }

    fileTraining.release();

    // Training
    CvSVMParams param = CvSVMParams();

    param.svm_type = CvSVM::C_SVC;
    param.kernel_type = CvSVM::RBF; //CvSVM::RBF, CvSVM::LINEAR ...
    param.degree = 0; // for poly
    param.gamma = 20; // for poly/rbf/sigmoid
    param.coef0 = 0; // for poly/sigmoid

    param.C = 7; // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
    param.nu = 0.0; // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
    param.p = 0.0; // for CV_SVM_EPS_SVR

    param.class_weights = NULL; // for CV_SVM_C_SVC
    param.term_crit.type = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS;
    param.term_crit.max_iter = 1000;
    param.term_crit.epsilon = 1e-6;

    svm.train_auto(trainingData, trainingClasses, cv::Mat(), cv::Mat(), param);

    cout << "Training complete." << endl;
}
