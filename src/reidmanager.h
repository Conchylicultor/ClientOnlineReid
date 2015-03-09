#ifndef REIDMANAGER_H
#define REIDMANAGER_H


#include <string>
#include <vector>

#include "features.h"

using namespace std;


enum class ReidMode {RELEASE, TRAINING, TESTING};


struct CamInfosElement
{
    size_t hashCodeCameraId;
    int beginDate;
    int endDate;
    cv::Vec2f entranceVectorOrigin;
    cv::Vec2f entranceVectorEnd;
    cv::Vec2f exitVectorOrigin;
    cv::Vec2f exitVectorEnd;
};

struct TransitionElement
{
    // Information on the first stage of the transition (leave the camera)
    size_t hashCodeCameraIdOut;
    cv::Vec2f exitVectorOrigin;
    cv::Vec2f exitVectorEnd;

    // Information on the final stage of the transition (reappearance)
    size_t hashCodeCameraIdIn;
    cv::Vec2f entranceVectorOrigin;
    cv::Vec2f entranceVectorEnd;

    int transitionDuration; // Can be negative if the person reappear in a camera before leaving the previous one
};

struct PersonElement
{
    vector<FeaturesElement> features;
    vector<CamInfosElement> camInfosList;
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

    void recordReceivedData(); // Just encapsulate the two following functions
    void recordTrainingSet();
    void recordTransitions();

    void testingTestingSet();
    void trainAndTestSet();

    void plotEvaluation();

    ReidMode currentMode;
    vector<PersonElement> database;

    vector<EvaluationElement> listEvaluation;// Evaluation which contain the datas to plot

    // TODO: Temporary ?
    // Move this code (and the declaration of the transition element) in a separate class (new Transition class or into the Feature class ?)
    // TODO: Cleanup all the transitions allusions in the Feature class (distance computation, record traning, )
    void plotTransitions();
    vector<TransitionElement> listTransitions;
};

#endif // REIDMANAGER_H
