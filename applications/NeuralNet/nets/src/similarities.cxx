
#include "../inc/similarities.h"

#include <cmath>

using namespace std;
using namespace nnet;


double SquaredEuclidianSim::operator()(double  *x,double  *y,unsigned int ndim){
    double out = 0;
    for(unsigned int i = 0; i < ndim; ++i){
        out += (x[i]-y[i])*(x[i]-y[i]);
    }
    return out;
}
void SquaredEuclidianSim::operator()(double **x,double *y,unsigned int ndim,
                                     unsigned int nevt,
                                     double *out){
    for(unsigned int i = 0; i < nevt; ++i){
        out[i] = (*this)(x[i], y, ndim);
    }
}
double SquaredEuclidianSim::operator()(vector<double> *x,vector<double> *y){
    double out = 0;
    for(unsigned int i = 0; i < x->size(); ++i){
        out += (x->at(i)-y->at(i))*(x->at(i)-y->at(i));
    }
    return out;
}
void  SquaredEuclidianSim::operator()(vector<vector<double> > *x,vector<double> *y,
                                      vector<double> *out){
    for(unsigned int i = 0; i < x->size(); ++i){
        out->at(i) = (*this)(&x->at(i), y);
    }
}



//========================================================================================


double EuclidianSim::operator()(double  *x, double  *y,unsigned int ndim){
    return sqrt(SquaredEuclidianSim::operator()(x, y, ndim));
}
void EuclidianSim::operator()(double **x,double *y,unsigned int ndim,
                              unsigned int nevt,double *out){
    for(unsigned int i = 0; i < nevt; ++i){
        out[i] = (*this)(x[i], y, ndim);
    }
}
double EuclidianSim::operator()(vector<double> *x,vector<double> *y){
    return sqrt(SquaredEuclidianSim::operator()(x, y));
}
void  EuclidianSim::operator()(vector<vector<double> > *x,vector<double> *y,
                                      vector<double> *out){
    for(unsigned int i = 0; i < x->size(); ++i){
        out->at(i) = (*this)(&x->at(i), y);
    }
}

