#include "features.h"

#include <fstream>

// Variables for features computation
#define HIST_SIZE 100

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

float Features::computeDistance(const FeaturesElement &elem1, const FeaturesElement &elem2)
{
    const int dimentionFeatureVector = 3 // Histogram
                                     + NB_MAJOR_COLORS_KEEP; // Major colors
    Mat rowFeatureVector = cv::Mat::ones(1, dimentionFeatureVector, CV_32FC1);

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

    // Feature Scaling

    return svm.predict(rowFeatureVector);
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

void Features::loadMachineLearning()
{
    // Loading file
    FileStorage fileTraining("../../Data/Received/training.yml", FileStorage::READ);

    if(!fileTraining.isOpened())
    {
        cout << "Error: cannot open the training file" << endl;
        exit(0);
    }

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
