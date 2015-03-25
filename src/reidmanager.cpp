#include "reidmanager.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <ctime>
#include <list>

using namespace std;

static const float thresholdValueSamePerson = 0.21;

struct ElemTraining
{
    unsigned int idPers1; // index in the dataset
    unsigned int valuePers1; // feature of the person
    unsigned int idPers2;
    unsigned int valuePers2;

    int same; // Positive or negative sample
};

ReidManager::ReidManager() :
    calibrationActive(false)
{
    Features::getInstance(); // Initialize the features (train the svm,...)
    Transition::getInstance(); // Same for the transitions (camera list,...)
    std::srand ( unsigned ( std::time(0) ) );
    namedWindow("MainWindow", WINDOW_NORMAL);

    setMode(ReidMode::RELEASE); // Default mode (call after initialized that loadMachineLearning has been set in case of TRAINING)
    setDebugMode(true); // Record the result
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

    SequenceElement currentSequence;

    size_t offset = 2; // The other ofsets values are extracted on the next function
    Transition::getInstance().extractArray(&arrayReceived[offset], offset, currentSequence.camInfo); // Has to be call before the feature extraction (modify the offset value)
    Features::getInstance().extractArray(&arrayReceived[offset], sizeArray-offset, currentSequence.features);

    delete arrayReceived;

    if(currentMode == ReidMode::TRAINING) // For computing or evaluate the result of our binary classifier
    {
        // Check if there is a new camera
        Transition::getInstance().checkCamera(currentSequence.camInfo);

        // We simply add the person to the dataset

        bool newPers = true; // Not found yet

        for(PersonElement &currentPers : database)
        {
            if(currentPers.hashId == hashSeqId || calibrationActive) // If calibration mode is activated, we concider there is just one person into the camera
            {
                currentPers.features.insert(currentPers.features.end(), currentSequence.features.begin(), currentSequence.features.end());
                currentPers.camInfoList.push_back(currentSequence.camInfo);
                newPers = false;
                break; // We only add the person once !
            }
        }

        // No match
        if(newPers)
        {
            // Add the new person to the database
            database.push_back(PersonElement());
            database.back().features.swap(currentSequence.features); // Can be done because a new person has an empty list
            database.back().camInfoList.push_back(currentSequence.camInfo);
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
                for(FeaturesElement featuresSequence : currentSequence.features)
                {
                    meanPrediction += Features::getInstance().predict(featuresDatabase, featuresSequence);
                }
            }
            meanPrediction /= (currentPers.features.size() * currentSequence.features.size());

            // Check the transition
            //Transition::getInstance().predict();

            // Match. Update database ?

            bool match = meanPrediction > thresholdValueSamePerson;
            bool matchError = false; // For debugging

            if(match)
            {
                cout << "Match (" << meanPrediction << ") : " << currentPers.hashId;
                newPers = false;

                if(currentMode == ReidMode::TESTING)
                {
                    if(currentPers.hashId != hashSeqId) // False positive
                    {
                        cout << " <<< ERROR";

                        matchError = true;

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

                    matchError = true;

                    listEvaluation.back().nbError++;
                    listEvaluation.back().nbErrorFalseNegativ++;
                    nbErrorClone++;
                }
                cout << endl;
            }

            // Record the results for checking
            if(debugMode)
            {
                plotDebugging(currentSequence, currentPers, match, matchError);
            }
        }

        // No match
        if(newPers)
        {
            cout << "No match: Add the new person to the dataset" << endl;

            // Add the new person to the database
            database.push_back(PersonElement());
            database.back().features.swap(currentSequence.features); // Can be swapped because the new person has an empty list
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
    else if(key == 'a' && currentMode == ReidMode::TRAINING)
    {
        cout << "Switch calibrations mode..." << endl;
        calibrationActive = !calibrationActive;
        if(calibrationActive)
        {
            cout << "Calibration mode activated!" << endl;
        }
        else
        {
            cout << "Calibration mode desactivated!" << endl;
        }
        cout << "Done" << endl;
    }
    else if(key == 'p' && currentMode == ReidMode::TRAINING)
    {
        cout << "Plot the transitions..." << endl;
        Transition::getInstance().plotTransitions();
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
    else if(key == 'd' && (currentMode == ReidMode::TESTING || currentMode == ReidMode::RELEASE))
    {
        cout << "Switch debug mode..." << endl;
        setDebugMode(!debugMode);
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
        Transition::getInstance().clearCameraMap();
    }
    else if(currentMode == ReidMode::TESTING)
    {
        cout << "testing";
    }
    cout << endl;
    // TODO: Clear database ?
    // TODO: Reload svm ?
}

void ReidManager::setDebugMode(bool newMode)
{
    debugMode = newMode;
    if(debugMode)
    {
        // Clear the folders
        system("exec rm ../../Data/Debug/Results/Difference/*");
        system("exec rm ../../Data/Debug/Results/Difference_errors/*");
        system("exec rm ../../Data/Debug/Results/Recognition/*");
        system("exec rm ../../Data/Debug/Results/Recognition_errors/*");
        cout << "Debug mode activated!" << endl;
    }
    else
    {
        cout << "Debug mode desactivated!" << endl;
    }
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

    fileTraining.release();

    Transition::getInstance().saveCameraMap();
}

void ReidManager::recordTransitions()
{
    cout << "Record transitions" << endl;

    vector<vector<CamInfoElement> > listSequencePerson;

    for(const PersonElement &currentPerson : database)
    {
        listSequencePerson.push_back(currentPerson.camInfoList);
    }

    Transition::getInstance().recordTransitions(listSequencePerson);
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
    // (No training for the transitions)

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

void ReidManager::plotDebugging(SequenceElement sequence, PersonElement person, bool same, bool error)
{
    // Don't save if there is no error
    if(currentMode == ReidMode::TESTING && !error)
    {
        return;
    }

    // Choose the destination

    string filenameDebug = "../../Data/Debug/Results/";

    if(error && same)
    {
        filenameDebug += "Recognition_errors/";
    }
    else if(error && !same)
    {
        filenameDebug += "Difference_errors/";
    }
    else if(same)
    {
        filenameDebug += "Recognition/";
    }
    else
    {
        filenameDebug += "Difference/";
    }

    // Extract the sub images

    vector<vector<Mat> > imageRows;
    imageRows.push_back(vector<Mat>(sequence.features.size()*2));
    imageRows.push_back(vector<Mat>(person.features.size()*2));

    size_t i = 0;
    size_t j = 0;

    int debugImgSize [2][2] = {0}; // [row][width, height]

    bool secondRow = false;
    for(size_t compteurTot = 0 ; compteurTot < sequence.features.size() + person.features.size() ; ++compteurTot)
    {
        // Check if second row
        if(j == 0 && compteurTot == sequence.features.size())
        {
            secondRow = true;
            i = 0;
            ++j;
        }

        const FeaturesElement *currentFeatureElem = nullptr;

        // Choose the current feature element
        if(!secondRow)
        {
            currentFeatureElem = &sequence.features.at(i);
        }
        else
        {
            currentFeatureElem = &person.features.at(i);
        }

        string filenameImage = "../../Data/Traces/"
                + to_string(currentFeatureElem->clientId) + "_"
                + to_string(currentFeatureElem->silhouetteId) + "_"
                + to_string(currentFeatureElem->imageId);

        Mat currentImage = imread(filenameImage + ".png");
        Mat currentImageMask = imread(filenameImage + "_mask.png");

        if(!currentImage.data || !currentImageMask.data)
        {
            cout << "Warning: cannot open images for debugging: " << filenameImage << endl;
            return;
        }

        imageRows.at(j).at(2*i) = currentImage;
        imageRows.at(j).at(2*i+1) = currentImageMask;

        debugImgSize[j][0] += currentImage.cols*2;
        debugImgSize[j][1] = std::max(debugImgSize[j][1], currentImage.rows);

        ++i;
    }

    // Compute the debug image

    Mat imgDebug(Size(std::max(debugImgSize[0][0], debugImgSize[1][0]),
                      debugImgSize[0][1] + debugImgSize[1][1]),
                 CV_8UC3, Scalar(200, 200, 200));

    int currentX = 0;
    int currentY = 0;

    for(i = 0 ; i < imageRows.size() ; ++i)
    {
        currentX = 0;
        for(j = 0 ; j < imageRows.at(i).size() ; ++j)
        {

            imageRows.at(i).at(j).copyTo(imgDebug(Rect(currentX, currentY, imageRows.at(i).at(j).cols, imageRows.at(i).at(j).rows)));
            currentX += imageRows.at(i).at(j).cols;
        }
        currentY += debugImgSize[i][1];
    }

    // Record

    imwrite(filenameDebug +
            to_string(sequence.features.front().clientId) + "_" + to_string(sequence.features.front().silhouetteId) +
            "_to_" + person.name + ".png", imgDebug);

    // Free the memory

    for(vector<Mat> &currentRow : imageRows)
    {
        for(Mat &currentImg : currentRow)
        {
            currentImg.release();
        }
    }
}
