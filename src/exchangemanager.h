#ifndef EXCHANGEMANAGER_H
#define EXCHANGEMANAGER_H

#include <list>
#include <string>
#include "mosquittopp.h"

struct ConnectedClient
{
    unsigned int id;
    unsigned int version_number;
};

class ExchangeManager : public mosqpp::mosquittopp
{
public:
    static ExchangeManager &getInstance();

private:
    ExchangeManager();
    ~ExchangeManager();

    // Assure C++ compatibility, handle errors, remove unused parameter
    void publish(const std::string &topic, int payloadlen=0, const void *payload=NULL, int qos=0, bool retain=false, int *mid = NULL);
    void subscribe(const std::string &sub, int qos=0);

    void on_message(const struct mosquitto_message *message);
    void onNewConnection(const struct mosquitto_message *message);
    void onRemovedConnection(const struct mosquitto_message *message);
    void onDataReceived(const struct mosquitto_message *message);
    void on_publish(int mid);

    std::list<ConnectedClient> listConnectedClients;

    // When exiting, we assure the broker receive the disconnection message
    int midExit;
    bool exitSend;
};

#endif // EXCHANGEMANAGER_H
