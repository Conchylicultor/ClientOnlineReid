#include "reidmanager.h"

#include <iostream>
#include <fstream>
#include <list>

using namespace std;

ReidManager::ReidManager()
{

}

void ReidManager::computeNext()
{
    string nextSeqString = getNextSeqString();
    if(nextSeqString.empty())
    {
        return;
    }

    cout << "Compute: " << nextSeqString << endl;
    ifstream seqFile("../../Data/Received/seq" + nextSeqString + ".txt", ios_base::in);
    if(!seqFile.is_open())
    {
        cout << "Unable to open the sequence file (please, check your working directory)" << endl;
        return;
    }

    size_t arrayReceivedSize = 0;
    float *arrayReceived = nullptr;

    seqFile >> arrayReceivedSize;

    arrayReceived = new float[arrayReceivedSize];


    for(size_t i = 0 ; i < arrayReceivedSize ; ++i)
    {
        seqFile >> arrayReceived[i];
        cout << arrayReceived[i] << endl;
    }

    delete arrayReceived;
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
