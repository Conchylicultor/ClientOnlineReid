#include "reidmanager.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <ctime>
#include <list>

using namespace std;
using namespace cv;

static const bool sequenceDatasetMode = false; // If true, all sequences will be added to the dataset as individual person

static const bool transitionsIncluded = false; // If true, the program will use the network topology information
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
    std::srand ( unsigned ( std::time(0) ) );
    namedWindow("MainWindow", WINDOW_AUTOSIZE);

    Features::getInstance(); // Initialize the features (train the svm,...)
    Transition::getInstance(); // Same for the transitions (camera list,...)

    listEvaluation.push_back(EvaluationElement{0,0,0,0,0,0,0,0,0}); // Origin for the evaluation

    setDebugMode(false); // Record the result
    setMode(ReidMode::RELEASE); // Default mode (call after initialized that loadMachineLearning has been set in case of TRAINING)
    updateGui();
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
                currentPers.sequenceList.push_back(SequenceElement());
                //currentPers.sequenceList.back().features.assign(currentSequence.features.begin(), currentSequence.features.end());
                currentPers.sequenceList.back().camInfo = currentSequence.camInfo;

                // We insert instead of pushing back, so we would have only one list of FeaturesElement (more easy to select the traning sample)
                currentPers.sequenceList.front().features.insert(
                    currentPers.sequenceList.front().features.begin(),
                    currentSequence.features.begin(),
                    currentSequence.features.end());

                newPers = false;
                break; // We only add the person once !
            }
        }

        // No match
        if(newPers)
        {
            // Add the new person to the database
            database.push_back(PersonElement());

            database.back().sequenceList.push_back(SequenceElement());
            database.back().sequenceList.back().features.swap(currentSequence.features);
            database.back().sequenceList.back().camInfo = currentSequence.camInfo;

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
            newEvalElement.nbCumulativeSuccess = listEvaluation.back().nbCumulativeSuccess;
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

        for(size_t currentPersId = 0 ; currentPersId < database.size() ; currentPersId++)
        {
            PersonElement &currentPers = database.at(currentPersId);

            // Check the visual similarity
            float similarityScore = 0.0;
            size_t nbFeaturesDatabase = 0;
            for(const SequenceElement &sequenceDatabase : currentPers.sequenceList) // Comparaison of each sequence
            {
                for(const FeaturesElement &featuresDatabase : sequenceDatabase.features)
                {
                    for(const FeaturesElement &featuresSequence : currentSequence.features)
                    {
                        similarityScore += Features::getInstance().predict(featuresDatabase, featuresSequence); // TODO: Instead of averaging all score, maybe it's better to have one different score for each sequence
                    }
                    nbFeaturesDatabase++;
                }
            }
            similarityScore /= (nbFeaturesDatabase * currentSequence.features.size());

            // Check the transition probability
            float transitionScore = 0.0;
            if(transitionsIncluded)
            {
                transitionScore = Transition::getInstance().predict(currentSequence.camInfo, currentPers.sequenceList.back().camInfo);
            }


            if(sequenceDatasetMode)
            {
                if(similarityScore > -0.8) // Record only significant edge
                {
                    listEdge.push_back({static_cast<float>(currentPersId + 1), // Vertex Id (Ids start at 1)
                                        static_cast<float>(database.size() + 1), // Vertex Id (will be added just after)
                                        static_cast<float>(similarityScore + 1.0)}); // Weigth (>0)
                }
            }
            else
            {
                // Match. Update database ?

                bool match = similarityScore * 1.0 + transitionScore * 1.0 > thresholdValueSamePerson; // TODO: Balance the weigth between
                bool matchError = false; // For debugging

                if(match)
                {
                    cout << "Match (" << similarityScore << ") : " << currentPers.hashId;
                    newPers = false;

                    // We update the informations on the current sequence
                    currentPers.sequenceList.push_back(SequenceElement());
                    currentPers.sequenceList.back().features = currentSequence.features;
                    currentPers.sequenceList.back().camInfo = currentSequence.camInfo;

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
                            listEvaluation.back().nbCumulativeSuccess++;
                            isRecognizeOnce = true; // At least once
                        }
                    }

                    cout << endl;
                }
                else if(currentMode == ReidMode::TESTING)
                {
                    cout << "Diff (" << similarityScore << ")";

                    if (currentPers.hashId == hashSeqId) // False negative
                    {
                        cout << " <<< ERROR";

                        matchError = true;

                        listEvaluation.back().nbError++;
                        listEvaluation.back().nbErrorFalseNegativ++;
                        nbErrorClone++;
                    }
                    else
                    {
                        listEvaluation.back().nbCumulativeSuccess++;
                    }
                    cout << endl;
                }

                // Record the results for checking
                if(debugMode)
                {
                    plotDebugging(currentSequence, currentPers, match, matchError);
                }
            }
        }

        // No match
        if(newPers || sequenceDatasetMode) // If we are in sequence mode, we add all sequences
        {
            cout << "No match: Add the new person to the dataset" << endl;

            // Add the new person to the database
            database.push_back(PersonElement());

            database.back().sequenceList.push_back(SequenceElement());
            database.back().sequenceList.back().features.swap(currentSequence.features); // Can be swapped because the new person has an empty list
            database.back().sequenceList.back().camInfo = currentSequence.camInfo;

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
    else if(key == 'n' && currentMode == ReidMode::RELEASE)
    {
        cout << "Record the network..." << endl;
        recordNetwork();
        cout << "Done" << endl;
    }
    else if(key == 'q')
    {
        cout << "Exit..." << endl;
        return true;
    }

    if(key != -1)
    {
        updateGui();
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

void ReidManager::updateGui()
{
    Mat monitorScreen = Mat::zeros(480, 640, CV_8UC3);

    String mainText;
    String secondaryText;
    String instructionLine1Text;
    String instructionLine2Text;

    Point positionText(15, 30);
    Scalar color(255, 255, 255);

    putText(monitorScreen, "----- Re-identification -----", positionText, FONT_HERSHEY_SIMPLEX, 0.6, color);
    positionText.y += 30;

    putText(monitorScreen, "Controls: Exit (q), Switch mode (s),...", positionText, FONT_HERSHEY_SIMPLEX, 0.5, color);
    positionText.y += 30;

    switch (currentMode) {
    case ReidMode::TRAINING:
        mainText = "Training (no recognition, used to train the svm as binary classifier)";

        if(calibrationActive)
        {
            secondaryText = "Calibration active (all incoming sequences are considered as the same person)";
        }
        else
        {
            secondaryText = "Calibration mode disable";
        }

        instructionLine1Text = "Create training set (t), test with incoming sequences (g) or do both (b)";
        instructionLine2Text = "Switch calibration (a) or not, then record (c) or plot (p) the transitions";
        break;
    case ReidMode::TESTING:
        mainText = "Testing (all incoming sequences have to be labelized)";

        instructionLine1Text = "Plot the evaluation (e)";

        break;
    case ReidMode::RELEASE:
        mainText = "Release (no labelized sequences)";

        break;
    default:
        break;
    }

    if(currentMode == ReidMode::TESTING || currentMode == ReidMode::RELEASE)
    {
        if(sequenceDatasetMode)
        {
            secondaryText += "Graph mode on (each incoming sequence is recorded as new person)";
        }
        else
        {
            secondaryText = "Graph mode off";
        }

        if(debugMode)
        {
            secondaryText += ", Debug mode on";
        }
        else
        {
            secondaryText += ", Debug mode off";
        }

        instructionLine2Text = "Save the recognition graph (n) ; Switch to debug mode (d) (save all images)";
    }

    putText(monitorScreen, "Mode: " + mainText, positionText, FONT_HERSHEY_SIMPLEX, 0.5, color);
    positionText.y += 15;
    putText(monitorScreen, "Mode option: " + secondaryText, positionText, FONT_HERSHEY_SIMPLEX, 0.5, color);
    positionText.y += 30;

    putText(monitorScreen, "Instructions:", positionText, FONT_HERSHEY_SIMPLEX, 0.5, color);
    positionText.y += 15;
    putText(monitorScreen, instructionLine1Text, positionText, FONT_HERSHEY_SIMPLEX, 0.5, color);
    positionText.y += 15;
    putText(monitorScreen, instructionLine2Text, positionText, FONT_HERSHEY_SIMPLEX, 0.5, color);
    positionText.y += 15;

    imshow("MainWindow", monitorScreen);
    waitKey(1); // update immediately
}

void ReidManager::selectPairs(Mat &dataSet, Mat &classesSet)
{
    vector<ElemTraining> listDataSet;

    unsigned int idPers1 = 0;
    for(PersonElement currentPerson : database)
    {
        unsigned int nbFeatures = currentPerson.sequenceList.front().features.size();
        unsigned int nbSample = nbFeatures * 2; // Arbitrary number

        // Positive samples
        for(unsigned int i = 0 ; i < nbSample ; ++i)
        {
            unsigned int value1 = std::rand() % nbFeatures;
            unsigned int value2 = std::rand() % nbFeatures;

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
                unsigned int value1 = std::rand() % nbFeatures;
                unsigned int value2 = std::rand() % database.at(idPers2).sequenceList.front().features.size();

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
        Features::getInstance().computeDistance(database.at(currentSetElem.idPers1).sequenceList.front().features.at(currentSetElem.valuePers1),
                                                database.at(currentSetElem.idPers2).sequenceList.front().features.at(currentSetElem.valuePers2),
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
}

void ReidManager::recordTransitions()
{
    cout << "Record transitions" << endl;

    vector<vector<CamInfoElement>> listSequencePerson;

    for(const PersonElement &currentPerson : database)
    {
        listSequencePerson.push_back(vector<CamInfoElement>());
        for(const SequenceElement &currentSequence : currentPerson.sequenceList)
        {
            listSequencePerson.back().push_back(currentSequence.camInfo);
        }
    }

    Transition::getInstance().recordTransitions(listSequencePerson);

    Transition::getInstance().saveCameraMap();
}

void ReidManager::recordNetwork()
{
    ofstream fileNetwork("../../Data/OutputReid/network.net");
    if(!fileNetwork.is_open())
    {
        cout << "Error: cannot open the network file (Is the working directory correct or the directory \"OutputReid/\" created ?)" << endl;
        return;
    }

    int indexVertex = 0;
    for(const PersonElement &person : database)
    {
        for(size_t j = 0 ; j < person.sequenceList.size() ; ++j)
        {
            indexVertex++; // Count the number of vertex
        }
    }

    fileNetwork << "*Vertices " << indexVertex << endl;

    indexVertex = 1;
    for(const PersonElement &person : database)
    {
        for(size_t j = 0 ; j < person.sequenceList.size() ; ++j)// Each person only contain one sequence if sequenceDatasetMode is active
        {
            const SequenceElement &sequence = person.sequenceList.at(j);

            fileNetwork << indexVertex << " \"";

            // Sequence id
            string fileId = to_string(sequence.features.front().clientId) + "_"
                          + to_string(sequence.features.front().silhouetteId);
            fileNetwork << " seq:" << fileId;

            // The date (for the filters)
            if(transitionsIncluded)
            {
                fileNetwork << " date:" << sequence.camInfo.beginDate;
            }

            // The person id
            if(currentMode == ReidMode::TESTING || !sequenceDatasetMode)
            {
                fileNetwork << " pers:" << person.hashId;
                if(j != 0)
                {
                    listEdge.push_back({static_cast<float>(indexVertex-1), // Previous vertex Id
                                        static_cast<float>(indexVertex), // Current vertex Id
                                        1.0});
                }
                else if(person.sequenceList.size() > 1) // Connect the last sequence to the first
                {
                    listEdge.push_back({static_cast<float>(indexVertex), // Current vertex Id
                                        static_cast<float>(indexVertex + person.sequenceList.size() - 1), // Last vertex Id
                                        1.0});
                }
            }

            fileNetwork  << "\"" << endl;
            indexVertex++;
        }
    }

    fileNetwork << "*Edges" << endl;

    for(array<float, 3> currentEdge : listEdge)
    {
        fileNetwork << currentEdge.at(0) << " " << currentEdge.at(1) << " " << currentEdge.at(2) << endl;
    }

    fileNetwork.close();
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
        int posLegend = 10;

        pt1.y = windowsEvalHeight - stepVerticalAxis * evalElemPrev.nbPersonAdded;
        pt2.y = windowsEvalHeight - stepVerticalAxis * evalElemNext.nbPersonAdded;
        color = Scalar(255, 0, 0);
        putText(imgEval, "Person added", Point(10, posLegend), FONT_HERSHEY_SIMPLEX, 0.4, color);
        line(imgEval, pt1, pt2, color);
        posLegend += 10;

        pt1.y = windowsEvalHeight - stepVerticalAxis * evalElemPrev.nbError;
        pt2.y = windowsEvalHeight - stepVerticalAxis * evalElemNext.nbError;
        color = Scalar(0, 0, 255);
        putText(imgEval, "Errors (Cumulative)", Point(10, posLegend), FONT_HERSHEY_SIMPLEX, 0.4, color);
        line(imgEval, pt1, pt2, color);
        posLegend += 10;

        pt1.y = windowsEvalHeight - stepVerticalAxis * (evalElemNext.nbError - evalElemPrev.nbError);
        pt2.y = windowsEvalHeight - stepVerticalAxis * (evalElemNext.nbError - evalElemPrev.nbError);
        color = Scalar(0, 0, 255);
        putText(imgEval, "Errors", Point(10, posLegend), FONT_HERSHEY_SIMPLEX, 0.4, color);
        line(imgEval, pt1, pt2, color);
        posLegend += 10;

        pt1.y = windowsEvalHeight - stepVerticalAxis * evalElemPrev.nbCumulativeSuccess;
        pt2.y = windowsEvalHeight - stepVerticalAxis * evalElemNext.nbCumulativeSuccess;
        color = Scalar(0, 255, 0);
        putText(imgEval, "Success (Cumulative)", Point(10, posLegend), FONT_HERSHEY_SIMPLEX, 0.4, color);
        line(imgEval, pt1, pt2, color);
        posLegend += 10;

        pt1.y = windowsEvalHeight - stepVerticalAxis * (evalElemNext.nbCumulativeSuccess - evalElemPrev.nbCumulativeSuccess);
        pt2.y = windowsEvalHeight - stepVerticalAxis * (evalElemNext.nbCumulativeSuccess - evalElemPrev.nbCumulativeSuccess);
        color = Scalar(0, 255, 0);
        putText(imgEval, "Success", Point(10, posLegend), FONT_HERSHEY_SIMPLEX, 0.4, color);
        line(imgEval, pt1, pt2, color);
        posLegend += 10;

        pt1.y = windowsEvalHeight - stepVerticalAxis * evalElemPrev.nbErrorFalseNegativ;
        pt2.y = windowsEvalHeight - stepVerticalAxis * evalElemNext.nbErrorFalseNegativ;
        color = Scalar(0, 255, 255);
        putText(imgEval, "False negativ", Point(10, posLegend), FONT_HERSHEY_SIMPLEX, 0.4, color);
        line(imgEval, pt1, pt2, color);
        posLegend += 10;

        pt1.y = windowsEvalHeight - stepVerticalAxis * evalElemPrev.nbErrorFalsePositiv;
        pt2.y = windowsEvalHeight - stepVerticalAxis * evalElemNext.nbErrorFalsePositiv;
        color = Scalar(0, 130, 255);
        putText(imgEval, "False positive", Point(10, posLegend), FONT_HERSHEY_SIMPLEX, 0.4, color);
        line(imgEval, pt1, pt2, color);
        posLegend += 10;

        /*pt1.y = windowsEvalHeight - stepVerticalAxis * evalElemPrev.nbErrorWithoutClone;
        pt2.y = windowsEvalHeight - stepVerticalAxis * evalElemNext.nbErrorWithoutClone;
        color = Scalar(115, 32, 150);
        putText(imgEval, "Without clone", Point(10, posLegend), FONT_HERSHEY_SIMPLEX, 0.4, color);
        line(imgEval, pt1, pt2, color);
        posLegend += 10;

        pt1.y = windowsEvalHeight - stepVerticalAxis * evalElemPrev.nbClone;
        pt2.y = windowsEvalHeight - stepVerticalAxis * evalElemNext.nbClone;
        color = Scalar(73, 92, 17);
        putText(imgEval, "Clones", Point(10, posLegend), FONT_HERSHEY_SIMPLEX, 0.4, color);
        line(imgEval, pt1, pt2, color);
        posLegend += 10;*/
    }

    // Display
    namedWindow("Evaluation Results", CV_WINDOW_AUTOSIZE);
    imshow("Evaluation Results", imgEval);
}

void ReidManager::plotDebugging(const SequenceElement &sequence, const PersonElement &person, bool same, bool error)
{
    // Don't save if there is no error
    if(currentMode == ReidMode::TESTING && !error)
    {
        return;
    }

    // Choose the destination

    string filenameDebug = "../../Data/Debug/Results/";

    if(same)
    {
        filenameDebug += "Recognition";
    }
    else
    {
        filenameDebug += "Difference";
    }

    if(error)
    {
        filenameDebug += "_errors";
    }

    filenameDebug += "/";

    // Extract the sub images

    const SequenceElement &sequencePerson = person.sequenceList.front();

    vector<vector<Mat>> imageRows;
    imageRows.push_back(vector<Mat>(sequence.features.size()*2));
    imageRows.push_back(vector<Mat>(sequencePerson.features.size()*2));

    size_t i = 0;
    size_t j = 0;

    int debugImgSize [2][2] = {0}; // [row][width, height]

    bool secondRow = false;
    for(size_t compteurTot = 0 ; compteurTot < sequence.features.size() + sequencePerson.features.size() ; ++compteurTot)
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
            currentFeatureElem = &sequencePerson.features.at(i);
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
