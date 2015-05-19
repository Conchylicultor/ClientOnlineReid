#ifndef REIDMANAGER_H
#define REIDMANAGER_H


#include <string>
#include <vector>
#include <array>

#include "features.h"
#include "transition.h"


enum class ReidMode {RELEASE, TRAINING, TESTING};

struct SequenceElement
{
    std::vector<FeaturesElement> features;
    CamInfoElement camInfo;
};

struct PersonElement
{
    std::vector<SequenceElement> sequenceList;
    std::string name;
    size_t hashId;
};

struct EvaluationElement
{
    // X Datas
    int nbSequence;

    // Y Datas
    int nbError;
    int nbCumulativeSuccess; // Nb of sequence correctly matched

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
    std::string getNextSeqString() const;
    float *reconstructArray(const std::string &seqId, size_t &sizeOut) const;

    void setMode(const ReidMode &newMode);
    void setDebugMode(bool newMode);

    void selectPairs(cv::Mat &dataSet, cv::Mat &classesSet);

    void recordReceivedData(); // Just encapsulate the two following functions (recordTrainingSet and recordTransition)
    void recordTrainingSet();
    void recordTransitions();

    void recordNetwork();

    void testingTestingSet();
    void trainAndTestSet();

    void plotEvaluation();
    void plotDebugging(const SequenceElement &sequence, const PersonElement &person, bool same, bool error=false); // Save the results on disk

    ReidMode currentMode;
    bool calibrationActive;
    bool debugMode; // Save the images for checking the recognition
    std::vector<PersonElement> database;

    std::vector<EvaluationElement> listEvaluation;// Evaluation which contain the datas to plot

    // Network
    std::vector<std::array<float,3>> listEdge; // Index of the vertex and weigth
};

#endif // REIDMANAGER_H
