/**
* W. H. Bell
*
* A list of classes Root does not understand by default and have to
* incorporated into a dictionary.
*/

#include <vector>
#include <map>
#include <string>
#ifdef __MAKECINT__
#pragma extra_include "vector";
#pragma extra_include "map";
//#pragma link C++ class std::vector<bool>+;
//#pragma link C++ class std::vector<short>+;
//#pragma link C++ class std::vector<int>+;
//#pragma link C++ class std::vector<long>+;
//#pragma link C++ class std::vector<double>+;
#pragma link C++ class std::vector<std::vector<unsigned int> >+;
#pragma link C++ class std::vector<std::vector<int> >+;
#pragma link C++ class std::vector<std::vector<double> >+;
#pragma link C++ class std::vector<std::vector<double> >+;
#pragma link C++ class std::vector<std::vector<string> >+;
#pragma link C++ class std::map<std::string,bool>+;

#pragma link C++ defined_in "../../nets/inc/neuralnet.h";
#pragma link C++ defined_in "../../nets/inc/backpropagation.h";
#pragma link C++ defined_in "../../nets/inc/rprop.h";
#pragma link C++ defined_in "../../nets/inc/pruning.h";
#pragma link C++ defined_in "../../nets/inc/pruning_minw.h";
#pragma link C++ defined_in "../../nets/inc/pruning_neurons.h";
#pragma link C++ defined_in "../../nets/inc/saturated.h";
#pragma link C++ defined_in "../../infra/inc/iomgrroot_pattern.h";
#pragma link C++ defined_in "../../infra/inc/iomgr.h";
#pragma link C++ defined_in "../../train/inc/trninfo.h";
#pragma link C++ defined_in "../../train/inc/trainbp.h";
#pragma link C++ defined_in "../../train/inc/trainpso.h";
#pragma link C++ defined_in "../../train/inc/crossval.h";
#pragma link C++ defined_in "../../train/inc/pso.h";
#pragma link C++ defined_in "../../function/functions.h";

#endif
