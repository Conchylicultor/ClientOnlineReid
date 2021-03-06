#include "exchangemanager.h"

#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv.hpp"

using namespace std;

#define CLIENTID    "ReidClient"

const string protocolVersionTopic = "reid_client/protocol_version";
const string newCamTopic = "cam_clients/new_connection";
const string removeCamTopic = "cam_clients/remove_connection";
const string dataCamTopic = "cam_clients/data";

const unsigned int version_protocol = 1;
const unsigned int max_version_protocol = 1000000;

ExchangeManager &ExchangeManager::getInstance()
{
    static ExchangeManager instance;
    return instance;
}

ExchangeManager::ExchangeManager() :
    mosqpp::mosquittopp(),
    midExit(0),
    exitSend(false)
{
    // Clear the received file
    ofstream receivedFile("../../Data/Received/received.txt", ofstream::out | ofstream::trunc);
    receivedFile.close();

    // Initialise the library
    mosqpp::lib_init();

    // Loading the configuration
    string brokerAdress;

    cv::FileStorage fileConfig("../config.yml", cv::FileStorage::READ);
    if(!fileConfig.isOpened())
    {
        cout << "Error: cannot open the configuration file" << endl;
        exit(0);
    }

    fileConfig["brokerIp"] >> brokerAdress;

    fileConfig.release();


    // Creating a client instance
    // /!\ Warning: Each client must have a UNIQUE id !!!
    this->reinitialise(CLIENTID, true);

    // Configure the will before the connection
    int result = this->will_set(protocolVersionTopic.c_str(),
                   sizeof(unsigned int),
                   &max_version_protocol,
                   2,                                 // QoS important
                   true);                             // Retained
    switch (result)
    {
    case MOSQ_ERR_SUCCESS:
        break;
    case MOSQ_ERR_INVAL:
        cout << "Error testimony : invalid parameters" << endl;
        break;
    case MOSQ_ERR_NOMEM:
        cout << "Error testimony : out of memory" << endl;
        break;
    default:
        cout << "Error testimony : ???" << endl;
        break;
    }

    // Connect the client to the broker.
    cout << "Try connecting the reidentification client to the brocker..." << endl;
    result = this->connect(brokerAdress.c_str());
    // Check the result
    switch (result)
    {
    case MOSQ_ERR_SUCCESS:
        cout << "Connection successful" << endl;
        break;
    case MOSQ_ERR_INVAL:
        cout << "Error connection : invalid parameters" << endl;
        break;
    case MOSQ_ERR_ERRNO:
        cout << "Error connection : server error" << endl;
        break;
    default:
        cout << "Error connection : ???" << endl;
        break;
    }

    // Subscription to search for new connected clients
    this->subscribe(newCamTopic, 2);
    this->subscribe(removeCamTopic, 2);
    this->subscribe(dataCamTopic, 2);

    // Publish a retained message for the new clients which will connect to this clients
    this->publish(protocolVersionTopic,
                  sizeof(unsigned int),              // We send only one value
                  &version_protocol,                 // Current version
                  2,                                 // QoS
                  true);                             // Retained
}

ExchangeManager::~ExchangeManager()
{
    // Send the last will
    this->publish(protocolVersionTopic,
                  sizeof(unsigned int),
                  &max_version_protocol,
                  2,
                  true,
                  &midExit);

    // Wait that the message has been send
    while(!exitSend)
    {
        loop();
    }

    // Disconnect properly
    this->disconnect();

    mosqpp::lib_cleanup(); // End of use of this library
}

void ExchangeManager::publish(const string &topic, int payloadlen, const void *payload, int qos, bool retain, int *mid)
{
    // Publish
    int result = mosqpp::mosquittopp::publish(mid, topic.c_str(), payloadlen, payload, qos, retain);

    // Check the result
    switch (result) {
    case MOSQ_ERR_SUCCESS:
        cout << "Publication successful to : " << topic << endl;
        break;
    case MOSQ_ERR_INVAL:
        cout << "Error publication : invalid parameters" << endl;
        break;
    case MOSQ_ERR_NOMEM:
        cout << "Error publication : out of memory" << endl;
        break;
    case MOSQ_ERR_NO_CONN:
        cout << "Error publication : client not connected to a broker" << endl;
        break;
    case MOSQ_ERR_PROTOCOL:
        cout << "Error publication : protocol error" << endl;
        break;
    case MOSQ_ERR_PAYLOAD_SIZE:
        cout << "Error publication : payloadlen is too large" << endl;
        break;
    default:
        cout << "Error publication : ???" << endl;
        break;
    }
}

void ExchangeManager::subscribe(const std::string &sub, int qos)
{
    int result = mosqpp::mosquittopp::subscribe(NULL, sub.c_str(), qos);

    // Check the result
    switch (result) {
    case MOSQ_ERR_SUCCESS:
        cout << "Subscription successful : " << sub << endl;
        break;
    case MOSQ_ERR_INVAL:
        cout << "Error publication : invalid parameters" << endl;
        break;
    case MOSQ_ERR_NOMEM:
        cout << "Error publication : out of memory" << endl;
        break;
    case MOSQ_ERR_NO_CONN:
        cout << "Error publication : client not connected to a broker" << endl;
        break;
    default:
        cout << "Error publication : ???" << endl;
        break;
    }
}

void ExchangeManager::on_message(const mosquitto_message *message)
{
    cout << "Message received : " << message->topic << endl;
    if(message->topic == newCamTopic)
    {
        onNewConnection(message);
    }
    else if(message->topic == removeCamTopic)
    {
        onRemovedConnection(message);
    }
    else if(message->topic == dataCamTopic)
    {
        onDataReceived(message);
    }
}

void ExchangeManager::onNewConnection(const mosquitto_message *message)
{
    cout << "New camera connected..." << endl;

    // Error if we don't send the same type of data or if the architectures are incompatible
    if(message->payloadlen != sizeof(ConnectedClient))
    {
        cout << "Error: Impossible to add the camera, data sent incompatible" << endl;
        return;
    }

    // Add the new camera to the list
    listConnectedClients.push_back(*(ConnectedClient*)message->payload);
    cout << listConnectedClients.back().id << endl;
}

void ExchangeManager::onRemovedConnection(const mosquitto_message *message)
{
    cout << "Delete cam..." << endl;
    cout << *(unsigned int*)message->payload << endl;

    size_t beforeSize = listConnectedClients.size();

    listConnectedClients.remove_if( [&message](ConnectedClient &currentClient) -> bool {
        if(currentClient.id == *(unsigned int*)message->payload)
        {
            return true;
        }
        return false;
    });

    if(beforeSize == listConnectedClients.size())
    {
        cout << "Warning: client not found" << endl;
    }
}

void ExchangeManager::onDataReceived(const mosquitto_message *message)
{
    // Update the received index

    fstream receivedFile("../../Data/Received/received.txt", ios::out | ios::in | ios::app);
    if(!receivedFile.is_open())
    {
        cout << "Unable to open the received file (please, check your working directory)" << endl;
    }

    // Read current file
    int currentIndex = 0;
    string ligne;
    while(getline(receivedFile, ligne))
    {
        if(!ligne.empty())
        {
            currentIndex = stoi(ligne);
        }
    }
    //receivedFile.unget();
    receivedFile.clear();
    currentIndex++;
    receivedFile << currentIndex << endl;

    receivedFile.close();

    // Save on disk the received data

    ofstream featuresFile("../../Data/Received/seq" + std::to_string(currentIndex) + ".txt", ios_base::out);

    float *receivedArray = reinterpret_cast<float*>(message->payload);
    unsigned int receivedSize = message->payloadlen/sizeof(float);

    featuresFile << receivedSize << endl; // Write the size for the reader
    for (unsigned int i = 0 ; i < receivedSize ; ++i) // Write the data
    {
        // We convert the value in int to avoid loss when we wrote the data
        featuresFile << reinterpret_cast<int&>(receivedArray[i]) << endl;
    }

    featuresFile.close();
}

void ExchangeManager::on_publish(int mid)
{
    // If the exit message has been send, we can exit properly
    if(mid == midExit)
    {
        exitSend = true;
    }
}
