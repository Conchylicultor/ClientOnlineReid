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

struct EvaluationElement
{
    // X Datas
    int nbSequence;

    // Y Datas
    int nbError;
    int nbSuccess; // Final result

    int nbErrorFalsePositiv;// << Database corrupted
    int nbErrorFalseNegativ;

    int nbErrorWithoutClone;// Errors exept if match at least recognize (at least) by one of it's clone in the dataset
    int nbErrorPersonAdded;// When some is added but is already in the datset
    int nbClone;// When some is added but is already in the datset

    int nbPersonAdded;// Infos
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

    void setMode(const ReidMode &newMode);

    void selectPairs(Mat &dataSet, Mat &classesSet);
    void recordTrainingSet();
    void testingTestingSet();

    void plotEvaluation();

    ReidMode currentMode;
    vector<PersonElement> database;

    vector<EvaluationElement> listEvaluation;// Evaluation which contain the datas to plot
};

#endif // REIDMANAGER_H
