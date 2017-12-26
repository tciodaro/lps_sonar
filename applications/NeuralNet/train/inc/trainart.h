
#ifndef TRAINARTH
#define TRAINARTH

#include "../../nets/inc/artnet.h"
#include "../../io/inc/iomgr.h"
#include "trninfo.h"
#include <string>
#include <vector>

namespace nnet{
    class Trainart{
        public:
            Trainart();
            virtual ~Trainart();

            virtual bool initialize();
            /// Train a network
            virtual bool train();
            virtual void recolor(std::vector<std::vector<double> > &data,
                                 std::vector<std::vector<double> > &target);
            /// Train parameters
            double trn_eta;
            double trn_eta_decay;
            double trn_mem_n_it; // Total allowed iterations with no change
            unsigned int trn_nshow;
            double trn_tol; // Allowed tolerance to consider stall
            unsigned int trn_max_it; // number of iterations
            unsigned int trn_max_stall; // Max number of stalled iterations
            unsigned int trn_max_no_new_neuron; // Max number of iterations with nothing new
            double trn_max_neurons_rate; // Control max number of neurons
            bool trn_phase1;
            bool trn_phase2;
            bool trn_phase3;
            double trn_initial_radius;
            double trn_radius_factor;
            // Performance
            std::vector<double> trn_perf;
            std::vector<double> val_perf;
            std::vector<double> tst_perf;

            double trn_accuracy;
            double trn_sp;
            double val_accuracy;
            double val_sp;
            double tst_accuracy;
            double tst_sp;

            std::string trn_opt_radius_strategy;
            std::string trn_class_strategy;


            ARTNet *get_art(){return m_artnet;};
            void    set_art(ARTNet *art){m_artnet = art;};

            TrnInfo_Pattern  *get_trninfo(){return m_trninfo;};
            IOMgr           &get_iomgr(){return (*m_iomgr);};

            void set_iomgr(IOMgr *mgr){m_iomgr = mgr;};
        protected:
            void performance_calculator(std::vector<std::vector<double> > *data,
                                        std::vector<std::vector<double> > *target,
                                        std::vector<unsigned int> *indexes,
                                        std::vector<unsigned int> &classes,
                                        std::vector<double> &perf_per_class,
                                        double &sp,
                                        double &accuracy);
            void getClasses(std::vector<unsigned int> &v,
                            std::vector<std::vector<double> >*tgt);

            double optimalRadius(std::vector<unsigned int> *itrn);
            double optimalRadius_DistMode(std::vector<double> &D);
            double optimalRadius_Percentile(std::vector<double> &D);
            double optimalRadius_STD(std::vector<double> &D);

            bool train_phase1(double train_class);
            bool train_phase2(double train_class);
            void performance();

            TrnInfo_Pattern *m_trninfo;
            IOMgr *m_iomgr;
            ARTNet *m_artnet;

    };
}

#endif

