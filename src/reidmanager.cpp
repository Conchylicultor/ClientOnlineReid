#include "reidmanager.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <ctime>
#include <list>

using namespace std;

static const float thresholdValueSamePerson = 0.21;

static const int transitionDurationMin = -10;
static const int transitionDurationMax = 10; // Duration limits of the transition

struct ElemTraining
{
    unsigned int idPers1; // index in the dataset
    unsigned int valuePers1; // feature of the person
    unsigned int idPers2;
    unsigned int valuePers2;

    int same; // Positive or negative sample
};

ReidManager::ReidManager()
{
    Features::getInstance(); // Initialize the features (train the svm,...)
    std::srand ( unsigned ( std::time(0) ) );
    namedWindow("MainWindow", WINDOW_NORMAL);

    setMode(ReidMode::TRAINING); // Default mode (call after initialized that loadMachineLearning has been set in case of TRAINING)
    listEvaluation.push_back(EvaluationElement{0,0,0,0,0,0,0,0,0}); // Origin
}

void ReidManager::computeNext()
{
    // Get the next received sequence
    string nextSeqString = getNextSeqString();
    if(nextSeqString.empty())
    {
        return;
    }

    cout << "Compute sequence: " << nextSeqString << endl;

    size_t sizeArray = 0;
    float *arrayReceived = reconstructArray(nextSeqString, sizeArray);

    // Extractions on the features

    size_t hashSeqId = reconstructHashcode(&arrayReceived[0]); // Get the id of the sequence

    vector<FeaturesElement> listCurrentSequenceFeatures;
    CamInfosElement currentSequenceCamInfos;
    size_t offset = 2; // The other ofsets values are extracted on the next function
    Features::getInstance().extractArray(&arrayReceived[offset], sizeArray-offset, listCurrentSequenceFeatures);
    // TODO Cleanup: Merge CameraInfos and list<FeaturesElement> into a struct SequenceElem
    // TODO Cleanup: Get instance Transitions and extract camera infos ; Remove folowing code:
    FeaturesElement extractCamInfos = listCurrentSequenceFeatures.front();
    currentSequenceCamInfos.hashCodeCameraId = extractCamInfos.hashCodeCameraId;
    currentSequenceCamInfos.beginDate = extractCamInfos.beginDate;
    currentSequenceCamInfos.endDate = extractCamInfos.endDate;
    currentSequenceCamInfos.entranceVectorOrigin = extractCamInfos.entranceVectorOrigin;
    currentSequenceCamInfos.entranceVectorEnd = extractCamInfos.entranceVectorEnd;
    currentSequenceCamInfos.exitVectorOrigin = extractCamInfos.exitVectorOrigin;
    currentSequenceCamInfos.exitVectorEnd = extractCamInfos.exitVectorEnd;
    // End Removing

    delete arrayReceived;

    if(currentMode == ReidMode::TRAINING) // For computing or evaluate the result of our binary classifier
    {
        // Check if there is a new camera
        Features::getInstance().checkCamera(listCurrentSequenceFeatures.front());

        // We simply add the person to the dataset

        bool newPers = true; // Not found yet

        for(PersonElement &currentPers : database)
        {
            if(currentPers.hashId == hashSeqId)
            {
                currentPers.features.insert(currentPers.features.end(), listCurrentSequenceFeatures.begin(), listCurrentSequenceFeatures.end());
                currentPers.camInfosList.push_back(currentSequenceCamInfos);
                newPers = false;
            }
        }

        // No match
        if(newPers)
        {
            // Add the new person to the database
            database.push_back(PersonElement());
            database.back().features.swap(listCurrentSequenceFeatures);
            database.back().camInfosList.push_back(currentSequenceCamInfos);
            database.back().name = std::to_string(hashSeqId);
            database.back().hashId = hashSeqId;
        }
    }
    else if(currentMode == ReidMode::RELEASE || currentMode == ReidMode::TESTING)
    {
        // For evaluation
        bool alreadyInDataset = false;
        bool isRecognizeOnce = false;
        int nbErrorClone = 0;
        if(currentMode == ReidMode::TESTING)
        {
            // Evaluation which contain the datas to plot
            EvaluationElement newEvalElement;
            // X
            newEvalElement.nbSequence = listEvaluation.size() + 1;
            // Y: Cumulative results
            newEvalElement.nbError             = listEvaluation.back().nbError;
            newEvalElement.nbSuccess           = listEvaluation.back().nbSuccess;
            newEvalElement.nbErrorFalsePositiv = listEvaluation.back().nbErrorFalsePositiv;
            newEvalElement.nbErrorFalseNegativ = listEvaluation.back().nbErrorFalseNegativ;
            newEvalElement.nbErrorPersonAdded  = listEvaluation.back().nbErrorPersonAdded;
            newEvalElement.nbErrorWithoutClone = listEvaluation.back().nbErrorWithoutClone;
            newEvalElement.nbClone             = listEvaluation.back().nbClone;
            newEvalElement.nbPersonAdded       = listEvaluation.back().nbPersonAdded;
            listEvaluation.push_back(newEvalElement);

            for(PersonElement currentPers : database)
            {
                if(currentPers.hashId == hashSeqId)
                {
                    alreadyInDataset = true;
                }
            }
        }

        // Match with the dataset

        bool newPers = true; // Not recognize yet

        for(PersonElement currentPers : database)
        {
            float meanPrediction = 0.0;
            for(FeaturesElement featuresDatabase : currentPers.features)
            {
                for(FeaturesElement featuresSequence : listCurrentSequenceFeatures)
                {
                    meanPrediction += Features::getInstance().predict(featuresDatabase, featuresSequence);
                }
            }
            meanPrediction /= (currentPers.features.size() * listCurrentSequenceFeatures.size());

            // Match. Update database ?
            if(meanPrediction > thresholdValueSamePerson)
            {
                cout << "Match (" << meanPrediction << ") : " << currentPers.hashId;
                newPers = false;

                if(currentMode == ReidMode::TESTING)
                {
                    if(currentPers.hashId != hashSeqId) // False positive
                    {
                        cout << " <<< ERROR";

                        listEvaluation.back().nbError++;
                        listEvaluation.back().nbErrorFalsePositiv++;
                        listEvaluation.back().nbErrorWithoutClone++;
                    }
                    else
                    {
                        isRecognizeOnce = true; // At least once
                    }
                }

                cout << endl;
            }
            else if(currentMode == ReidMode::TESTING)
            {
                cout << "Diff (" << meanPrediction << ")";

                if (currentPers.hashId == hashSeqId) // False negative
                {
                    cout << " <<< ERROR";

                    listEvaluation.back().nbError++;
                    listEvaluation.back().nbErrorFalseNegativ++;
                    nbErrorClone++;
                }
                cout << endl;
            }
        }

        // No match
        if(newPers)
        {
            cout << "No match: Add the new person to the dataset" << endl;

            // Add the new person to the database
            database.push_back(PersonElement());
            database.back().features.swap(listCurrentSequenceFeatures);
            database.back().name = std::to_string(database.size());
            database.back().hashId = hashSeqId;

            if(currentMode == ReidMode::TESTING)
            {
                listEvaluation.back().nbPersonAdded++;
            }
        }

        if(currentMode == ReidMode::TESTING)
        {
            if(alreadyInDataset && !isRecognizeOnce)
            {
                listEvaluation.back().nbErrorWithoutClone += nbErrorClone;
            }

            if(alreadyInDataset && newPers)
            {
                listEvaluation.back().nbClone++;
            }
        }

    }
}

bool ReidManager::eventHandler()
{
    char key = waitKey(10);
    if(key == 's')
    {
        cout << "Switch mode..." << endl;
        if(currentMode == ReidMode::RELEASE)
        {
            setMode(ReidMode::TRAINING);
        }
        else if(currentMode == ReidMode::TRAINING)
        {
            setMode(ReidMode::TESTING);
        }
        else if(currentMode == ReidMode::TESTING)
        {
            setMode(ReidMode::RELEASE);
        }
    }
    else if(key == 't' && currentMode == ReidMode::TRAINING)
    {
        cout << "Creating the training set (from the received data)..." << endl;
        recordReceivedData();
        cout << "Done" << endl;
    }
    else if(key == 'c' && currentMode == ReidMode::TRAINING)
    {
        cout << "Calibrations of the cameras..." << endl;
        recordTransitions();
        cout << "Done" << endl;
    }
    else if(key == 'p' && currentMode == ReidMode::TRAINING)
    {
        cout << "Plot the transitions..." << endl;
        plotTransitions();
        cout << "Done" << endl;
    }
    else if(key == 'b' && currentMode == ReidMode::TRAINING)
    {
        cout << "Evaluate the learning algorithm..." << endl;
        trainAndTestSet();
        cout << "Done" << endl;
    }
    else if(key == 'g' && currentMode == ReidMode::TRAINING)
    {
        cout << "Testing the received data..." << endl;
        testingTestingSet();
        cout << "Done" << endl;
    }
    else if(key == 'e' && currentMode == ReidMode::TESTING)
    {
        cout << "Plot the evaluation data..." << endl;
        plotEvaluation();
        cout << "Done" << endl;
    }
    else if(key == 'q')
    {
        cout << "Exit..." << endl;
        return true;
    }
    return false;
}

string ReidManager::getNextSeqString() const
{
    string nextSeqString;

    // Read all lines
    ifstream receivedFileIn("../../Data/Received/received.txt", ios::in);
    if(!receivedFileIn.is_open())
    {
        cout << "Unable to open the received file (please, check your working directory)" << endl;
    }
    else
    {
        string line;
        list<std::string> filelines;
        while (receivedFileIn)
        {
             getline(receivedFileIn, line);
             if(!line.empty())
             {
                 filelines.push_back(line);
             }
        }
        receivedFileIn.close();

        // Extract the current element

        if(!filelines.empty())
        {
            nextSeqString = filelines.front();
            filelines.pop_front();
        }

        // Save the file
        ofstream receivedFileOut("../../Data/Received/received.txt", ios::out | ios::trunc);
        if(!receivedFileOut.is_open())
        {
            cout << "Unable to open the received file (please, check your working directory)" << endl;
        }

        for(string currentLine : filelines)
        {
            receivedFileOut << currentLine << endl;
        }

        receivedFileOut.close();
    }

    return nextSeqString;
}

float *ReidManager::reconstructArray(const string &seqId, size_t &sizeOut) const
{
    // Extraction of the received array
    ifstream seqFile("../../Data/Received/seq" + seqId + ".txt", ios_base::in);
    if(!seqFile.is_open())
    {
        cout << "Unable to open the sequence file (please, check your working directory)" << endl;
        return nullptr;
    }

    size_t arrayReceivedSize = 0;
    float *arrayReceived = nullptr;

    seqFile >> arrayReceivedSize;

    arrayReceived = new float[arrayReceivedSize];


    int tempValue = 0;
    for(size_t i = 0 ; i < arrayReceivedSize ; ++i)
    {
        seqFile >> tempValue;
        arrayReceived[i] = reinterpret_cast<float&>(tempValue);
    }

    sizeOut = arrayReceivedSize;
    return arrayReceived;
}

void ReidManager::setMode(const ReidMode &newMode)
{
    currentMode = newMode;
    cout << "Selected mode: ";
    if(currentMode == ReidMode::RELEASE)
    {
        cout << "release";
    }
    else if(currentMode == ReidMode::TRAINING)
    {
        cout << "training";
        // Clear the cameraMap (we learn from a clear)
        Features::getInstance().clearCameraMap();
    }
    else if(currentMode == ReidMode::TESTING)
    {
        cout << "testing";
    }
    cout << endl;
    // TODO: Clear database ?
    // TODO: Reload svm ?
}

void ReidManager::selectPairs(Mat &dataSet, Mat &classesSet)
{
    vector<ElemTraining> listDataSet;

    unsigned int idPers1 = 0;
    for(PersonElement currentPerson : database)
    {
        unsigned int nbSample = currentPerson.features.size() * 2; // Arbitrary number

        // Positive samples
        for(unsigned int i = 0 ; i < nbSample ; ++i)
        {
            unsigned int value1 = std::rand() % currentPerson.features.size();
            unsigned int value2 = std::rand() % currentPerson.features.size();

            if(value1 == value2)
            {
                --i;
            }
            else
            {
                listDataSet.push_back(ElemTraining());
                listDataSet.back().idPers1 = idPers1;
                listDataSet.back().idPers2 = idPers1;
                listDataSet.back().valuePers1 = value1;
                listDataSet.back().valuePers2 = value2;
                listDataSet.back().same = 1; // Positive sample
            }
        }

        // Negative samples
        for(unsigned int i = 0 ; i < nbSample ; ++i)
        {
            unsigned int idPers2 = std::rand() % database.size();

            if(idPers1 == idPers2)
            {
                --i;
            }
            else
            {
                unsigned int value1 = std::rand() % currentPerson.features.size();
                unsigned int value2 = std::rand() % database.at(idPers2).features.size();

                listDataSet.push_back(ElemTraining());
                listDataSet.back().idPers1 = idPers1;
                listDataSet.back().idPers2 = idPers2;
                listDataSet.back().valuePers1 = value1;
                listDataSet.back().valuePers2 = value2;
                listDataSet.back().same = -1; // Negative sample
            }
        }

        ++idPers1;
    }

    // Randomize
    std::random_shuffle(listDataSet.begin(), listDataSet.end());

    for(ElemTraining currentSetElem : listDataSet)
    {
        Mat rowFeatureVector;
        Features::getInstance().computeDistance(database.at(currentSetElem.idPers1).features.at(currentSetElem.valuePers1),
                                                database.at(currentSetElem.idPers2).features.at(currentSetElem.valuePers2),
                                                rowFeatureVector); // No scaling yet (wait that all data are received)

        Mat rowClass = cv::Mat::ones(1, 1, CV_32FC1);
        rowClass.at<float>(0,0) = currentSetElem.same;

        dataSet.push_back(rowFeatureVector);
        classesSet.push_back(rowClass);
    }
}

void ReidManager::recordReceivedData()
{
    // Create the training set
    recordTrainingSet();

    // Compute the transitions
    recordTransitions();
}

void ReidManager::recordTrainingSet()
{
    cout << "Record training set" << endl;

    if(database.size() <= 1)
    {
        cout << "Error: only one person in the database, cannot compute the training set" << endl;
        return;
    }

    Mat trainingData;
    Mat trainingClasses;

    // Create a training set (from the received data)
    selectPairs(trainingData, trainingClasses);

    if(trainingData.rows == 0)
    {
        cout << "Error: training set empty" << endl;
        return;
    }

    // Compute scaling factors
    Mat scaleFactors(1, trainingData.cols, CV_32FC1);

    // We compute the scaling factor for each dimention
    for(size_t i = 0 ; i < static_cast<unsigned>(trainingData.cols) ; ++i)
    {
        float maxValue = 0.0;
        for(size_t j = 0 ; j < static_cast<unsigned>(trainingData.rows) ; ++j)
        {
            float currentValue = trainingData.at<float>(j,i); // Accessing by (row,col)
            if(currentValue > maxValue)
            {
                maxValue = currentValue;
            }
        }

        // Warning if maxValue == 0
        if(maxValue < 0.00001) // Floating value => no strict equality
        {
            cout << "Warning: No scaling for the param " << i << endl;
            maxValue = 1.0; // No scaling in this case
        }

        scaleFactors.at<float>(0,i) = maxValue;
    }

    // Set scale factor to the feature
    Features::getInstance().setScaleFactors(scaleFactors);

    // Scale
    for(size_t i = 0 ; i < static_cast<unsigned>(trainingData.rows) ; ++i)
    {
        Features::getInstance().scaleRow(trainingData.row(i)); // Now the row is scaled
    }

    // Record the training data
    FileStorage fileTraining("../../Data/Training/training.yml", FileStorage::WRITE);
    if(!fileTraining.isOpened())
    {
        cout << "Error: Cannot record the training file (folder does not exist ?)" << endl;
    }

    fileTraining << "trainingData" << trainingData;
    fileTraining << "trainingClasses" << trainingClasses;
    fileTraining << "scaleFactors" << scaleFactors;

    Features::getInstance().saveCameraMap(fileTraining);

    fileTraining.release();
}

void ReidManager::recordTransitions()
{
    cout << "Record transitions" << endl;
    for(PersonElement const &currentPerson : database)
    {
        for(size_t i = 0 ; i < currentPerson.camInfosList.size() ; ++i)
        {
            // TODO: What is the best way to determine the transitions ? Can the algorithm be improved ?
            // Can we have multiple valid transitions for one sequence (not just between two cameras) ?

            // Looking for the smallest transition
            int closestCamInfos = -1;
            int closestCamInfosDuration = -1;
            for(size_t j = 0 ; j < currentPerson.camInfosList.size() ; ++j)
            {
                // Find the shortest transition
                if(i != j && currentPerson.camInfosList.at(i).beginDate < currentPerson.camInfosList.at(j).beginDate)
                {
                    int currentDuration = currentPerson.camInfosList.at(j).beginDate - currentPerson.camInfosList.at(i).beginDate; // > 0 (due to previous condition)

                    // First time
                    if(closestCamInfos == -1)
                    {
                        closestCamInfosDuration = currentDuration;
                        closestCamInfos = j;
                    }
                    else if(currentDuration < closestCamInfosDuration)
                    {
                        closestCamInfosDuration = currentDuration;
                        closestCamInfos = j;
                    }
                }
            }

            // Creation of the new transition
            TransitionElement newTransition;

            // The transition is between an exit and a re-entrance
            const CamInfosElement &camInfoElemtOut = currentPerson.camInfosList.at(i);
            newTransition.hashCodeCameraIdOut = camInfoElemtOut.hashCodeCameraId;
            newTransition.exitVectorOrigin = camInfoElemtOut.exitVectorOrigin;
            newTransition.exitVectorEnd = camInfoElemtOut.exitVectorEnd;

            // Match
            if(closestCamInfos != -1)
            {
                // The transition is between an exit and a re-entrance
                const CamInfosElement &camInfoElemtIn = currentPerson.camInfosList.at(closestCamInfos);
                newTransition.hashCodeCameraIdIn = camInfoElemtIn.hashCodeCameraId;
                newTransition.entranceVectorOrigin = camInfoElemtIn.entranceVectorOrigin;
                newTransition.entranceVectorEnd = camInfoElemtIn.entranceVectorEnd;

                newTransition.transitionDuration = camInfoElemtIn.beginDate - camInfoElemtOut.endDate; // != closestCamInfosDuration

                // Filter the transition if the duration is too long
                if(newTransition.transitionDuration > transitionDurationMax ||
                   newTransition.transitionDuration < transitionDurationMin)
                {
                    cout << "Transition too long(disappearance): " << newTransition.transitionDuration << endl;
                    closestCamInfos = -1; // Add the transition as disappearance transition
                }
            }

            // No match: disappearance
            if(closestCamInfos == -1)
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

void ReidManager::testingTestingSet()
{
    Mat testingData;
    Mat testingClasses;

    selectPairs(testingData, testingClasses);

    float successRate = 0.0;
    for(size_t i = 0 ; i < static_cast<unsigned>(testingData.rows) ; ++i)
    {
        // Test SVM
        float result = Features::getInstance().predictRow(testingData.row(i)); // Now the row is scaled

        // Compare result with the one expected
        if(result == testingClasses.at<float>(i))
        {
            successRate += 1.0;
        }
    }

    // Show the result
    if(testingData.rows > 0)
    {
        successRate /= testingData.rows;
        cout << "Success rate: " << successRate * 100.0 << "%" << endl;
    }
    else
    {
        cout << "Error: no data (database empty)" << endl;
    }
}

void ReidManager::trainAndTestSet()
{
    // Split the sequences into two set (We don't randomize the list: testing and training
    // set on two differents times)

    size_t const half_size = database.size() / 2;

    vector<PersonElement> trainDatabase (database.begin(), database.begin() + half_size);
    vector<PersonElement> testDatabase  (database.begin() + half_size, database.end());

    // Training
    std::swap(database, trainDatabase);
    recordReceivedData();
    std::swap(database, trainDatabase);

    // Retrain the classifier
    Features::getInstance().loadMachineLearning();

    // Testing
    cout << "Testing" << endl;
    std::swap(database, testDatabase);
    testingTestingSet();
    std::swap(database, testDatabase);
}

void ReidManager::plotEvaluation()
{
    const int stepHorizontalAxis = 20;
    const int stepVerticalAxis = 20;
    const int windowsEvalHeight = 900;

    Mat imgEval(Size(stepHorizontalAxis * listEvaluation.size(), windowsEvalHeight),
                CV_8UC3,
                Scalar(0,0,0));

    for(size_t i = 1 ; i < listEvaluation.size() ; ++i)
    {
        EvaluationElement evalElemPrev = listEvaluation.at(i-1);
        EvaluationElement evalElemNext = listEvaluation.at(i);

        Point pt1;
        Point pt2;
        pt1.x = stepHorizontalAxis * evalElemPrev.nbSequence;
        pt2.x = stepHorizontalAxis * evalElemNext.nbSequence;

        Scalar color;

        pt1.y = windowsEvalHeight - stepVerticalAxis * evalElemPrev.nbPersonAdded;
        pt2.y = windowsEvalHeight - stepVerticalAxis * evalElemNext.nbPersonAdded;
        color = Scalar(255, 0, 0);
        putText(imgEval, "Person added", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.4, color);
        line(imgEval, pt1, pt2, color);

        pt1.y = windowsEvalHeight - stepVerticalAxis * evalElemPrev.nbError;
        pt2.y = windowsEvalHeight - stepVerticalAxis * evalElemNext.nbError;
        color = Scalar(0, 255, 0);
        putText(imgEval, "Errors (Cumulativ)", Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.4, color);
        line(imgEval, pt1, pt2, color);

        pt1.y = windowsEvalHeight - stepVerticalAxis * evalElemPrev.nbErrorFalseNegativ;
        pt2.y = windowsEvalHeight - stepVerticalAxis * evalElemNext.nbErrorFalseNegativ;
        color = Scalar(0, 255, 255);
        putText(imgEval, "False negativ", Point(10, 40), FONT_HERSHEY_SIMPLEX, 0.4, color);
        line(imgEval, pt1, pt2, color);

        pt1.y = windowsEvalHeight - stepVerticalAxis * evalElemPrev.nbErrorFalsePositiv;
        pt2.y = windowsEvalHeight - stepVerticalAxis * evalElemNext.nbErrorFalsePositiv;
        color = Scalar(0, 130, 255);
        putText(imgEval, "False positiv", Point(10, 50), FONT_HERSHEY_SIMPLEX, 0.4, color);
        line(imgEval, pt1, pt2, color);

        pt1.y = windowsEvalHeight - stepVerticalAxis * evalElemPrev.nbErrorWithoutClone;
        pt2.y = windowsEvalHeight - stepVerticalAxis * evalElemNext.nbErrorWithoutClone;
        color = Scalar(115, 32, 150);
        putText(imgEval, "Without clone", Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.4, color);
        line(imgEval, pt1, pt2, color);

        pt1.y = windowsEvalHeight - stepVerticalAxis * evalElemPrev.nbClone;
        pt2.y = windowsEvalHeight - stepVerticalAxis * evalElemNext.nbClone;
        color = Scalar(73, 92, 17);
        putText(imgEval, "Clones", Point(10, 70), FONT_HERSHEY_SIMPLEX, 0.4, color);
        line(imgEval, pt1, pt2, color);

        pt1.y = windowsEvalHeight - stepVerticalAxis * (evalElemNext.nbError - evalElemPrev.nbError);
        pt2.y = windowsEvalHeight - stepVerticalAxis * (evalElemNext.nbError - evalElemPrev.nbError);
        color = Scalar(255, 255, 0);
        putText(imgEval, "Errors", Point(10, 10), FONT_HERSHEY_SIMPLEX, 0.4, color);
        line(imgEval, pt1, pt2, color);
    }

    // Display
    namedWindow("Evaluation Results", CV_WINDOW_AUTOSIZE);
    imshow("Evaluation Results", imgEval);
}

void ReidManager::plotTransitions()
{
    vector<Mat> camImgs(Features::getInstance().getCameraMap().size());
    for(Mat &currentImg : camImgs)
    {
        currentImg = Mat::zeros(Size(640,480),CV_8UC3);
    }

    for(TransitionElement const &currentTransition : listTransitions)
    {
        // Choose a random color for the arrow
        Scalar color;
        color[0] = std::rand() % 255;
        color[1] = std::rand() % 255;
        color[2] = std::rand() % 255;

        for(pair<int, size_t> currentCam : Features::getInstance().getCameraMap()) // For each camera
        {
            // Has an exit
            if(currentTransition.hashCodeCameraIdOut == currentCam.second)
            {
                Point pt1(currentTransition.exitVectorOrigin[0], currentTransition.exitVectorOrigin[1]);
                Point pt2(currentTransition.exitVectorEnd[0],    currentTransition.exitVectorEnd[1]);

                // Plot the arrow into the right cam
                cv::line(camImgs.at(currentCam.first), pt1, pt2, color);
                cv::circle(camImgs.at(currentCam.first), pt2, 2, color);
            }

            // Has an entrance
            if(currentTransition.hashCodeCameraIdIn == currentCam.second)
            {
                Point pt1(currentTransition.entranceVectorOrigin[0], currentTransition.entranceVectorOrigin[1]);
                Point pt2(currentTransition.entranceVectorEnd[0],    currentTransition.entranceVectorEnd[1]);

                // Plot the arrow into the right cam
                cv::line(camImgs.at(currentCam.first), pt1, pt2, color);
                cv::circle(camImgs.at(currentCam.first), pt2, 2, color);
            }
        }
    }

    for(pair<int, size_t> currentCam : Features::getInstance().getCameraMap()) // For each camera
    {
        cout << currentCam.first << endl;
        imshow("Transition: " + std::to_string(currentCam.second), camImgs.at(currentCam.first));
    }
}
