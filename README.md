ClientOnlineReid
================

## Introduction

Mqtt client which get features from other client and try to compute the reidentification

## Installation

**Dependencies:** In order to compile, the program will require QMake (tested with v3.0), OpenCv (tested with 2.4.8) and Mosquittopp (tested with v1.3.4). It also require a compiler compatible with C++11.

The path structure must look like this:
* _ReidPath/ClientOnlineReid/_ : This repository.
* _ReidPath/ClientOnlineReid/build/_ : Contain the executable file (**must be the working directory when running the program !**)
* _ReidPath/Data/Training_ : Contain the informations used to train the program. Those files can be generated in _training mode_. Otherwise, [here](https://gist.github.com/Conchylicultor/bde8de54f0adc44f2bb2) is an example of training file, if you don't want to generate yours.
* _ReidPath/Data/Received_ : Store the received sequences. This folder contain temporary files used by the program but useless for the final user (except eventually for debugging).
* _ReidPath/Data/OutputReid_ : Folder where the result of the reidentification will be stored.

In order to run the program, the folder ClientOnlineReid must contain a config.yml indicating the ip adress of the mqtt brocker:

```
%YAML:1.0
brokerIp:'192.168.100.13'
```

## Running

Multiple modes are proposed by this software:
* __Training mode__: the program record all person with there correct identity. No recognition is made. This mode is used to generate the differents positive and negative samples in order to train our binary classifier.
* __Testing mode__: Test our adaptative database and record the errors (kind, number,...). In this mode (as the previous one), the received sequences must have been labelized.
* __Release mode__: Same as training mode but without any kind of verification.

Here are the controls of the program:
* __s__: **S**witch between the modes (Release, Testing, Training)
* __q__: Exit the program
* For the training mode:
  * __t__: Generate and record positive and negative sample from the received data in order to generate the **T**raining set.
  * __g__: **G**enerate a testing set and test the efficiency of the binary classifier
  * __b__: Do **B**oth training and testing on the current received sequence
  * __a__: Activate the calibration mode. If the calibration mode is activated, all incoming sequences will be concidered as one person (no recognition possible but useful to see what path this person has done in order to compute the network topology). It is not require to be in calibration mode to compute the transitions.
  * __c__: Use the camera information of the received sequence to compute the transitions between the camera and having an idea **C**alibrate the camera. The results are saved into the _Training/_ folder.
  * __p__: Plot the different transitions between the camera which have been computed.
* For the testing mode:
  * __e__: In testing mode, **E**valuate and plot the result to show how well our algorithm perform. The data are also saved in a _.csv_ file (on the _OutputReid/_ directory) which can be imported in any spreadsheet software.
* For both testing and release:
  * __a__: Switch between the two recognition mode: either each sequence is added as a new person (whatever the person already exist in the dataset or not) or if there is a match (reidentification score above threshold), the sequence is only added to the most similar one.
  * __n__: Record the network (in the _OutputReid/_ directory) of the current recognition database.
  * __d__: Activate/desactivate the debug mode. If enable, all recognitions will be saved to precisely see where the recognition fails.

**Warning:** Each time the program save a file, it replace the previous file without asking for confirmation. Be carful if you don't want to lost data.
