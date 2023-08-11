//
//  BasisFunctionsSets.hpp
//  SparseFunctions
//
//  Created by Marco Battiato on 6/5/23.
//

#ifndef BasisFunctionsSets_hpp
#define BasisFunctionsSets_hpp

#include <Eigen/Dense>

namespace SparseFunction {

template <int order> struct TaylorBasis{
    template <typename Derived> static auto basisF(const Eigen::ArrayBase<Derived>& x){
        return x * TaylorBasis<order-1>::basisF(x);
    }
};

template <> struct TaylorBasis<0>{
    template <typename Derived> static auto basisF(const Eigen::ArrayBase<Derived>& x){
        return Eigen::ArrayBase<Derived>::Constant(x.rows(), x.cols(), 1);
    }
};

template <> struct TaylorBasis<1>{
    template <typename Derived> static auto basisF(const Eigen::ArrayBase<Derived>& x){
        return x.matrix().array();
    }
    
};



} // namespace SparseFunction

#endif /* BasisFunctionsSets_hpp */
