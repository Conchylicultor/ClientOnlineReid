ClientOnlineReid
================

Mqtt client which get features from other client and try to compute the reidentification

Here are the controls of the program:
* __s__: **S**witch mode (Release, Testing, Training)
* __t__: Generate and record positive and negative sample from the received data in order to generate the **T**raining set.
* __g__: **G**enerate a testing set and test the efficiency of the binary classifier
* __e__: In testing mode, **E**valuate and plot the result to show how well our algorithm perform

Multiple modes are proposed by this software:
* __Training mode__: the program record all person with there correct identity. No recognition is made. This mode is used to generate the differents positive and negative samples in order to train our binary classifier.
* __Testing mode__: Test our adaptative database and record the errors (kind, number,...)
* __Release mode__: Same as training mode but without any kind of verification.

In order to run the program, the folder ClientOnlineReid must contain a config.yml indicating the ip adress of the mqtt brocker:

```
%YAML:1.0
brokerIp:'192.168.100.13'
```
