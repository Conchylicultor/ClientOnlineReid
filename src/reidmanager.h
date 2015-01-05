#ifndef REIDMANAGER_H
#define REIDMANAGER_H


#include <string>

using namespace std;

class ReidManager
{
public:
    ReidManager();

    void computeNext();

private:
    string getNextSeqString() const;
};

#endif // REIDMANAGER_H
