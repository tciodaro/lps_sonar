


#ifndef SIMILARITIESH
#define SIMILARITIESH

#include <vector>
#include <string>

namespace nnet{
    // Basic structure of a sim_function
    class SimFunction{
        public:
            SimFunction(){name = "";};
            virtual double operator()(double  *x,
                                     double  *y,
                                     unsigned int ndim) = 0;
            virtual void  operator()(double **x,
                                     double *y,
                                     unsigned int ndim,
                                     unsigned int nevt,
                                     double *out) = 0;
            virtual double operator()(std::vector<double> *x,
                                     std::vector<double> *y) = 0;
            virtual void  operator()(std::vector<std::vector<double> > *x,
                                     std::vector<double> *y,
                                     std::vector<double> *out) = 0;

            std::string name;
    };

    // Inherited functions that actually do something

    /*  Squared Euclidian distance similarity.
     *
     *  The smaller the value, the grater is the similarity.
     */
    class SquaredEuclidianSim: public SimFunction{
        public:
            SquaredEuclidianSim() : SimFunction(){
                name = "sqeuclidian";
            };
            double operator()(double  *x,
                             double  *y,
                             unsigned int ndim);
            void  operator()(double **x,
                             double *y,
                             unsigned int ndim,
                             unsigned int nevt,
                             double *out);
            double operator()(std::vector<double> *x,
                             std::vector<double> *y);
            void  operator()(std::vector<std::vector<double> > *x,
                             std::vector<double> *y,
                             std::vector<double> *out);
    };


    /*  Euclidian distance similarity.
     *
     *  The smaller the value, the grater is the similarity.
     */
    class EuclidianSim: public SquaredEuclidianSim{
        public:
            EuclidianSim() : SquaredEuclidianSim(){
                name = "euclidian";
            };
            double operator()(double  *x,
                             double  *y,
                             unsigned int ndim);
            void  operator()(double **x,
                             double *y,
                             unsigned int ndim,
                             unsigned int nevt,
                             double *out);
            double operator()(std::vector<double> *x,
                             std::vector<double> *y);
            void  operator()(std::vector<std::vector<double> > *x,
                             std::vector<double> *y,
                             std::vector<double> *out);
    };



}
#endif

