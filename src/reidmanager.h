#ifndef REIDMANAGER_H
#define REIDMANAGER_H


#include <string>
#include <vector>

#include "features.h"

using namespace std;

struct PersonElement
{
    vector<FeaturesElement> features;
    string name;
};

class ReidManager
{
public:
    ReidManager();

    void computeNext();

private:
    string getNextSeqString() const;
    float *reconstructArray(const string &seqId, size_t &sizeOut) const;

    vector<PersonElement> database;
};

#endif // REIDMANAGER_H
