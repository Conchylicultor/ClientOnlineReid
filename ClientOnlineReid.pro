TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

QMAKE_CXXFLAGS += -std=c++11

SOURCES += src/main.cpp \
    src/exchangemanager.cpp \
    src/features.cpp \
    src/reidmanager.cpp

LIBS += -lmosquittopp
LIBS += `pkg-config opencv --libs`

HEADERS += \
    src/exchangemanager.h \
    src/features.h \
    src/reidmanager.h
