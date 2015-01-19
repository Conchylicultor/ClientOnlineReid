#include "reidmanager.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <ctime>
#include <list>

using namespace std;

static const float thresholdValueSamePerson = 0.5;

struct ElemTraining
{
    unsigned int idPers1; // index in the dataset
    unsigned int valuePers1; // feature of the person
    unsigned int idPers2;
    unsigned int valuePers2;

    int same; // Positive or negative sample
};

ReidManager::ReidManager() : currentMode(ReidMode::TRAINING)
{
    Features::getInstance(); // Initialize the features (train the svm,...)
    std::srand ( unsigned ( std::time(0) ) );
    namedWindow("MainWindow", WINDOW_NORMAL);
}

void ReidManager::computeNext()
{
    // Get the next received sequence
    string nextSeqString = getNextSeqString();
    if(nextSeqString.empty())
    {
        return;
    }

    cout << "Compute: " << nextSeqString << endl;

    size_t sizeArray = 0;
    float *arrayReceived = reconstructArray(nextSeqString, sizeArray);

    // Extractions on the features

    float hashSeqId = arrayReceived[0]; // Get the id of the sequence

    vector<FeaturesElement> listCurrentSequenceFeatures;
    size_t offset = 1;
    Features::getInstance().extractArray(&arrayReceived[offset], sizeArray-offset, listCurrentSequenceFeatures);

    delete arrayReceived;

    if(currentMode == ReidMode::TRAINING || currentMode == ReidMode::TESTING) // For computing or evaluate the result of our binary classifier
    {
        // We simply add the person to the dataset

        bool newPers = true; // Not found yet

        for(PersonElement currentPers : database)
        {
            if(currentPers.hashId == hashSeqId)
            {
                currentPers.features.insert(currentPers.features.end(), listCurrentSequenceFeatures.begin(), listCurrentSequenceFeatures.end());
                newPers = false;
            }
        }

        // No match
        if(newPers)
        {
            // Add the new person to the database
            database.push_back(PersonElement());
            database.back().features.swap(listCurrentSequenceFeatures);
            database.back().name = std::to_string(hashSeqId);
            database.back().hashId = hashSeqId;
        }
    }
    else if(currentMode == ReidMode::RELEASE)
    {
        // Match with the dataset

        bool newPers = true; // Not recognize yet

        for(PersonElement currentPers : database)
        {
            float meanPrediction = 0.0;
            for(FeaturesElement featuresDatabase : currentPers.features)
            {
                for(FeaturesElement featuresSequence : listCurrentSequenceFeatures)
                {
                    meanPrediction += Features::getInstance().computeDistance(featuresDatabase, featuresSequence);
                }
            }
            meanPrediction /= (currentPers.features.size() * listCurrentSequenceFeatures.size());

            // Match. Update database ?
            if(meanPrediction > thresholdValueSamePerson)
            {
                cout << "Match (" << meanPrediction << ") : " << currentPers.name << endl;
                newPers = false;
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
        }
    }
}

void ReidManager::eventHandler()
{
    char key = waitKey(10);
    if(key == 't')
    {
        cout << "Creating the training set (from the received data)..." << endl;
        recordTrainingSet();
        cout << "Done" << endl;
    }
    if(key == 'g')
    {
        cout << "Testing the received data..." << endl;
        testingTestingSet();
        cout << "Done" << endl;
    }
}

string ReidManager::getNextSeqString() const
{
    string nextSeqString;

    // Read all lines
    ifstream receivedFileIn("../../Data/Received/received.txt", ios::in);
    if(!receivedFileIn.is_open())
    {
        cout << "Unable to open the file (please, check your working directory)" << endl;
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
            cout << "Unable to open the file (please, check your working directory)" << endl;
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


    for(size_t i = 0 ; i < arrayReceivedSize ; ++i)
    {
        seqFile >> arrayReceived[i];
    }

    sizeOut = arrayReceivedSize;
    return arrayReceived;
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
                                                rowFeatureVector);

        Mat rowClass = cv::Mat::ones(1, 1, CV_32FC1);
        rowClass.at<float>(0,0) = currentSetElem.same;

        dataSet.push_back(rowFeatureVector);
        classesSet.push_back(rowClass);
    }
}

void ReidManager::recordTrainingSet()
{
    Mat trainingData;
    Mat trainingClasses;

    selectPairs(trainingData, trainingClasses);

    // Record the training data
    FileStorage fileTraining("../../Data/Training/training.yml", FileStorage::WRITE);

    fileTraining << "trainingData" << trainingData;
    fileTraining << "trainingClasses" << trainingClasses;

    fileTraining.release();
}

void ReidManager::testingTestingSet()
{
    Mat trainingData;
    Mat trainingClasses;

    selectPairs(trainingData, trainingClasses);

    float successRate = 0.0;
    for(size_t i = 0 ; i < static_cast<unsigned>(trainingData.rows) ; ++i)
    {
        // Test SVM
        float result = Features::getInstance().computeDistance(trainingData.row(i));

        // Compare result with expected
        if(result == trainingClasses.at<float>(i))
        {
            successRate += 1.0;
        }
    }

    // Show the result
    if(trainingData.rows > 0)
    {
        successRate /= trainingData.rows;
        cout << "Success rate: " << successRate * 100.0 << "%" << endl;
    }
    else
    {
        cout << "Error: no data (database empty)" << endl;
    }
}
