#include "reidmanager.h"

#include <iostream>
#include <fstream>
#include <list>

using namespace std;

static const float thresholdValueSamePerson = 0.5;

ReidManager::ReidManager()
{
    Features::getInstance();
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
