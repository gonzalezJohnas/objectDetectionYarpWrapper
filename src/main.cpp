#include <iostream>
#include "../include/iCub/ObjectDetectionModule.h"


using namespace yarp::os;
using namespace yarp::sig;


int main(int argc, char *argv[]) {

    Network yarp;
    ObjectDetectionModule module;

    ResourceFinder rf;
    rf.setVerbose(true);
    rf.setDefaultConfigFile("objectDetection.ini");      //overridden by --from parameter
    rf.setDefaultContext("objectDetection");              //overridden by --context parameter
    rf.configure(argc, argv);


    yInfo("resourceFinder: %s", rf.toString().c_str());

    module.runModule(rf);
    return 0;
}
