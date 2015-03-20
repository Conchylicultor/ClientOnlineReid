#ifndef REIDMANAGER_H
#define REIDMANAGER_H


#include <string>
#include <vector>

#include "features.h"
#include "transition.h"

using namespace std;


enum class ReidMode {RELEASE, TRAINING, TESTING};

struct SequenceElement
{
    vector<FeaturesElement> features;
    CamInfoElement camInfo;
};

struct PersonElement
{
    vector<FeaturesElement> features; // TODO: Replace those lines by vector<SequenceElement>
    vector<CamInfoElement> camInfoList;
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
    bool eventHandler();

private:
    string getNextSeqString() const;
    float *reconstructArray(const string &seqId, size_t &sizeOut) const;

    void setMode(const ReidMode &newMode);

    void selectPairs(Mat &dataSet, Mat &classesSet);

    void recordReceivedData(); // Just encapsulate the two following functions (recordTrainingSet and recordTransition)
    void recordTrainingSet();
    void recordTransitions();

    void testingTestingSet();
    void trainAndTestSet();

    void plotEvaluation();

    ReidMode currentMode;
    bool calibrationActive;
    vector<PersonElement> database;

    vector<EvaluationElement> listEvaluation;// Evaluation which contain the datas to plot
};

#endif // REIDMANAGER_H
