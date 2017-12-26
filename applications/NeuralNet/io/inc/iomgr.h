#ifndef IOMGRH
#define IOMGRH

#include <iostream>
#include <string>

#ifdef USEROOT
#include <TMatrixF.h>
#include <TTree.h>
#endif

#include "../../nets/inc/neuralnet.h"
#include "../../train/inc/trninfo.h"

namespace nnet{
    class IOMgr {
        public:
            // construtor
            IOMgr();
            // destructor
            virtual ~IOMgr();
            /// Get train/validation/test size
            unsigned int trn_size  (){return m_itrn.size();};
            unsigned int val_size  (){return m_ival.size();};
            unsigned int tst_size  (){return m_itst.size();};
            /// access to data
            virtual void   shuffle();
            /// access to data
            double *data(unsigned int it);
            double *target(unsigned int it);
            std::vector<std::vector<double> > *data  (){return m_data;};
            std::vector<std::vector<double> > *target(){return m_target;};
            /// Public access to train/validation/test indexes
            void set_trn(std::vector<unsigned int> &v);
            void set_val(std::vector<unsigned int> &v);
            void set_tst(std::vector<unsigned int> &v);
            std::vector<unsigned int> *get_trn(){return &m_itrn;};
            std::vector<unsigned int> *get_val(){return &m_ival;};
            std::vector<unsigned int> *get_tst(){return &m_itst;};
            /// Dimensions
            unsigned int  in_dim;
            unsigned int out_dim;
            unsigned int evt_dim;

            bool initialize(std::vector<std::vector<double> > *data,
                            std::vector<std::vector<double> > *target);
        protected:
            std::vector<std::vector<double> > *m_data;
            std::vector<std::vector<double> > *m_target;

            std::vector<unsigned int> m_itrn;
            std::vector<unsigned int> m_itst;
            std::vector<unsigned int> m_ival;
  };
} // namespace

#endif

// end of file
