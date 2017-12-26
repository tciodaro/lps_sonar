
%module neuralnet

%include "std_vector.i"
%include "std_string.i"

namespace std {
%template(RowF)  vector< double>;
%template(MatrixF) vector < vector< double> >;

%template(RowUI)  vector< unsigned int >;
%template(MatrixUI) vector < vector< unsigned int > >;

%template(RowI)  vector< int >;
%template(MatrixI) vector < vector< int > >;
}   


%typemap(out) nnet::TrnInfo *{
    std::string strtype = typeid(*$1).name(); // object type name   
    if(strtype.find("TrnInfo_Pattern") != std::string::npos){
        $result = SWIG_NewPointerObj(SWIG_as_voidptr($1), $descriptor(nnet::TrnInfo_Pattern *), $owner);
    }else if(strtype.find("TrnInfo") != std::string::npos){
        $result = SWIG_NewPointerObj(SWIG_as_voidptr($1), $descriptor(nnet::TrnInfo *), $owner);   
    }
}


%typemap(out) nnet::IOMgr *{
    std::string strtype = typeid(*$1).name(); // object type name   
    if(strtype.find("IOMgr_Pattern") != std::string::npos){
        $result = SWIG_NewPointerObj(SWIG_as_voidptr($1), $descriptor(nnet::IOMgr_Pattern *), $owner);
    }else if(strtype.find("IOMgr") != std::string::npos){
        $result = SWIG_NewPointerObj(SWIG_as_voidptr($1), $descriptor(nnet::IOMgr *), $owner);   
    }
}


%include "carrays.i"
%array_class(double, doubleArray);



