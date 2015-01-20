#ifndef REIDMANAGER_H
#define REIDMANAGER_H


#include <string>
#include <vector>

#include "features.h"

using namespace std;


enum class ReidMode {RELEASE, TRAINING, TESTING};

struct PersonElement
{
    vector<FeaturesElement> features;
    string name;
    size_t hashId;
};

class ReidManager
{
public:
    ReidManager();

    void computeNext();
    void eventHandler();

private:
    string getNextSeqString() const;
    float *reconstructArray(const string &seqId, size_t &sizeOut) const;

    void selectPairs(Mat &dataSet, Mat &classesSet);
    void recordTrainingSet();
    void testingTestingSet();

    ReidMode currentMode;
    vector<PersonElement> database;
};

#endif // REIDMANAGER_H
