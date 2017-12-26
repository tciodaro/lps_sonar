

#ifndef PYIOMGRH
#define PYIOMGRH


#include "iomgr.h"
#include "iomgr_pattern.h"
#include <vector>

namespace nnet{
    class PyIOMgr:  public IOMgr {
        public:
            PyIOMgr();
            virtual ~PyIOMgr();
            /// Load data to the manager
	        bool load(std::vector< std::vector<double> > &data, std::vector< std::vector<double> > &target);
    };
}


#endif


// end of file


