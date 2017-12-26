#ifndef TRNINFOH
#define TRNINFOH

#include <vector>
#include <string>

namespace nnet{
  //! Base info for the training
  class TrnInfo{
    public:
      /// Constructor
      TrnInfo();
      TrnInfo(const TrnInfo &trn);
      virtual void operator=(const TrnInfo &trn);
      virtual void copy(const TrnInfo *trn);
      /// Destructor
      virtual ~TrnInfo();
      // Extend structures
      virtual void resize(unsigned int n);
      // Clear structures
      virtual void clear();
      /// Monitors
      virtual void trn_monitor(unsigned int it, unsigned int N,
                               double *y, double *t, unsigned int K);
      virtual void val_monitor(unsigned int it, unsigned int N,
                               double *y, double *t, unsigned int K);
      virtual void tst_monitor(unsigned int it, unsigned int N,
                               double *y, double *t, unsigned int K);
      /// Is it a better epoch than the best one?
      virtual bool is_better(unsigned int iepoch, unsigned int min_epochs = 0);
      virtual bool is_better(TrnInfo *trn);
      virtual bool is_better(double perf);
      /// return true if p1 is better than p2
      virtual bool better(double p1, double p2) {return p1 < p2;}
      /// Resume calculations
      virtual void resume(unsigned int epoch){};
      /// Print epoch values
      virtual void print(unsigned int epoch);
      /// Calculate performance
      virtual double performance(double **, double **, unsigned int, unsigned int);
      virtual double performance(const char *type = "");
      double mse(); // non virtual
      virtual double perf_trn(unsigned int i){return mse_trn[i];};
      virtual double perf_val(unsigned int i){return mse_val[i];};
      virtual double perf_tst(unsigned int i){return mse_tst[i];};
      /// public attributesTrnInfo
      std::vector<double>    epoch;
      std::vector<double>    mse_trn;
      std::vector<double>    mse_val;
      std::vector<double>    mse_tst;
      unsigned int          bst_epoch;
      std::string           perfType;

      unsigned int getNVar(){return m_var_addrs.size();};
      std::vector<double> *getVarAddr(unsigned int i){return m_var_addrs[i];};
      const char *getVarName(unsigned int i){return m_var_names[i].c_str();};
      void copy_var(const char *, std::vector<double> &v);
      void copy_var(unsigned int ivar, std::vector<double> &v);
      const char *getName(){return m_name.c_str();};
      // Set epoch with values from TrnInfo best epoch
      void set_epoch(TrnInfo *p, unsigned int iepoch);

      static const char *kNAME;
    protected:
      std::string m_name;
      std::vector<std::string> m_var_names;
      std::vector<std::vector<double> *> m_var_addrs;
  };

    class TrnInfo_Pattern: public TrnInfo{
        public:
            /// Constructor
            TrnInfo_Pattern(unsigned int nclasses = 2);
            TrnInfo_Pattern(const TrnInfo_Pattern &trn);
            void operator=(const TrnInfo_Pattern &trn);
            void copy(const TrnInfo_Pattern *trn);
            /// Destructor
            virtual ~TrnInfo_Pattern();
            // Extend structures
            virtual void resize(unsigned int n);
            // Clear structures
            virtual void clear();
            /// Monitors
            virtual void trn_monitor(unsigned int it, unsigned int N,
                                     double *y, double *t, unsigned int K);
            virtual void val_monitor(unsigned int it, unsigned int N,
                                     double *y, double *t, unsigned int K);
            virtual void tst_monitor(unsigned int it, unsigned int N,
                                     double *y, double *t, unsigned int K);
            /// Is it a better epoch than the best one? SP
            virtual bool is_better(unsigned int iepoch, unsigned int min_epochs = 0);
            virtual bool is_better(TrnInfo *trn);
            virtual bool is_better(double perf);
            /// return true if p1 is better than p2
            virtual bool better(double p1, double p2) {return p1 > p2;}
            /// Resume calculations
            virtual void resume(unsigned int epoch);
            virtual void print(unsigned int epoch);
            /// Calculate performance
            virtual double performance(double **, double **, unsigned int, unsigned int);
            virtual double performance(const char *type = "");
            virtual double sp(); // get the best performance
            virtual double detection(); // average best detection efficiency
            virtual double false_alarm(); // average best false alarm
            virtual double perf_trn(unsigned int i){return sp_trn[i];};
            virtual double perf_val(unsigned int i){return sp_val[i];};
            virtual double perf_tst(unsigned int i){return sp_tst[i];};

            unsigned int get_number_of_classes(){return m_nclasses;};

            /// public attributes
            std::vector< std::vector<double> >   mse_trn_c; // for each class
            std::vector< std::vector<double> >   mse_val_c;
            std::vector< std::vector<double> >   mse_tst_c;
            std::vector< std::vector<double> >   tot_trn;
            std::vector< std::vector<double> >   tot_val;
            std::vector< std::vector<double> >   tot_tst;
            std::vector<double>                  sp_trn;
            std::vector<double>                  sp_val;
            std::vector<double>                  sp_tst;
            std::vector< std::vector<double> >   eff_trn;
            std::vector< std::vector<double> >   eff_val;
            std::vector< std::vector<double> >   eff_tst;
            std::vector< std::vector<double> >   fa_trn;
            std::vector< std::vector<double> >   fa_tst;
            std::vector< std::vector<double> >   fa_val;

            static const char *kNAME;

        protected:
            unsigned int m_nclasses;
            virtual void allocate();

    };

} // namespace




#endif

// end of file

