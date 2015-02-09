#include "features.h"

#include <fstream>

// Variables for features computation
#define HIST_SIZE 100

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
    const int dimentionFeatureVector = 3 // Histogram
                                     + NB_MAJOR_COLORS_KEEP; // Major colors
    rowFeatureVector = cv::Mat::ones(1, dimentionFeatureVector, CV_32FC1);

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

    // TODO: Add feature: camera id ; Add feature: time

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
    cv::Vec2f entranceVector(array[4], array[5]);
    cv::Vec2f exitVector(array[6], array[7]);

    // Shift the origin to remove the offset
    size_t offset = 8;
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
        currentElem.entranceVector = entranceVector;
        currentElem.exitVector = exitVector;

        // Histogram
        for(size_t channelId = 0 ; channelId < 3 ; ++channelId)
        {
            currentElem.histogramChannels.at(channelId).create(1,100,CV_32F); // TODO: Check order x,y !!!!!!!
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

    // Loading the scales factors
    fileTraining["scaleFactors"] >> scaleFactors;

    // Loading the scale training set
    Mat trainingData;
    Mat trainingClasses;
    fileTraining["trainingData"]    >> trainingData;
    fileTraining["trainingClasses"] >> trainingClasses;

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
