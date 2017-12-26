
// WORKS ONLY FOR 2 CLASSES

#ifndef IOMGRPATTERNH
#define IOMGRPATTERNH

#include "iomgr.h"
#include <vector>
#include <string>

namespace nnet{
  class IOMgr_Pattern: public IOMgr {
    public:
      // construtor
      IOMgr_Pattern();
      // destructor
      virtual ~IOMgr_Pattern();
      /// Data source
      void shuffle();
      void set_trn(std::vector<unsigned int> &trn);

      void complete_trn();

      virtual unsigned int trn_size  (){return m_used_itrn.size();};
      virtual std::vector<unsigned int> *get_trn(){return &m_used_itrn;};
      virtual void set_batch_size(unsigned int v);

      unsigned int get_trn(unsigned int cls, unsigned int i){return m_itrn_cls[cls][i];};

      unsigned int nclasses;

      std::string index_strategy;
    protected:

      std::vector<std::vector<unsigned int> > m_itrn_cls;
      std::vector<unsigned int> m_used_itrn;
  };
} // namespace

#endif
// end of file
