#ifndef REIDMANAGER_H
#define REIDMANAGER_H


#include <string>

#include "features.h"

using namespace std;

class ReidManager
{
public:
    ReidManager();

    void computeNext();

private:
    string getNextSeqString() const;
    float *reconstructArray(const string &seqId, size_t &sizeOut) const;
};

#endif // REIDMANAGER_H
