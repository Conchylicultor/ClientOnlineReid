#include <iostream>
#include <string>

#include "exchangemanager.h"
#include "reidmanager.h"

using namespace std;


int main()
{
    ExchangeManager &exchangeManager = ExchangeManager::getInstance();

    ReidManager reidentificationManager;

    while (1) {
        exchangeManager.loop();
        reidentificationManager.computeNext();
        reidentificationManager.eventHandler();
    }
    return 0;
}
