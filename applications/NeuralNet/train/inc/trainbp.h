
#ifndef TRAINBPH
#define TRAINBPH

#include <string>
#include <vector>
#include "../../nets/inc/backpropagation.h"
#include "../../io/inc/iomgr.h"
#include "trninfo.h"


namespace nnet{
  class Trainbp{
    public:
      /// Constructor
      Trainbp();
      /// Destructor
      virtual ~Trainbp();
      /// Initialize structures
      virtual bool initialize();
      /// Train a network
      virtual void train();
      /// Train parameters
      bool         fbatch;
      unsigned int nshow;
      unsigned int nepochs;
      unsigned int min_epochs;
      double        goal;
      unsigned int max_fail;
      unsigned int batch_size;
      std::string  net_task; // 'estimation', 'classification'

      void             set_net(Backpropagation *net){m_bpnet = net;};
      Backpropagation &get_net(){return (*m_bpnet);};
      TrnInfo         *get_trninfo(){return (m_trninfo);};
      IOMgr           &get_iomgr(){return (*m_iomgr);};

      void set_iomgr(IOMgr *mgr){m_iomgr = mgr;};

    protected:
      virtual void update_batch ();
      virtual void update_online();
      virtual void validate();
      virtual void test();
      virtual void show();

      /// Attributes
      unsigned int m_currEpoch;
      unsigned int m_nFails;

      TrnInfo *m_trninfo;

      IOMgr *m_iomgr;

      Backpropagation *m_bpnet; // official net

  }; // class
} // namespace


#endif

// end of file


