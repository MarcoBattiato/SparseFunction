//
//  SparseFunctionStruct.hpp
//  SparseFunctions
//
//  Created by Marco Battiato on 6/5/23.
//
//  We assume that we can make a spectral discretisation of a 1 variable function using basis functions of increasing order from 0 to a maxOrder
//  Notice that we assume here that there is a single basis function for each specific order. That is not always the case. This code does not
//  provide functionalities in that case.
//
//  The most straightforward way of generating basis functions for a multivariate case is to construct a Cartesian product of the single variable
//  basis functions. This is however wasteful and scales very badly with the number of dimensions.
//
//  We then construct basis functions for the multivariate case by taking products of univariate basis functions with a total order not exceeding
//  a predetermined maximum total order.
//
//  The simplest example is with polynomial basis functions. Let us assume that we use basis fucntions
//  f_0(x) = 1; f_1(x) = x; f_2(x) = x^2; f_3(x) = x^3;
//  where the subscript is the order
//
//  We can construct 2 variate basis functions as
//  g_00(x,y) = f_0(x)*f_0(y)
//  g_01(x,y) = f_0(x)*f_1(y);  g_10(x,y) = f_1(x)*f_0(y)
//  g_02(x,y) = f_0(x)*f_2(y);  g_11(x,y) = f_1(x)*f_1(y);  g_20(x,y) = f_2(x)*f_0(y)
//  etc
//
//  We include all the basis functions up to a maximum total order is fixed.
//
//  The object representing a discretised function has to contain all the coefficients for all the basis functions above.
//  The structure is built by nesting objects.
//  Let us suppose that we need to discretise a function of 3 variables with a maximum degree of 2
//  The top level will be of type
//     sparseFunctionStruct<totalOrder = 2, maxOrder = 2, nVariables = 3, basisFunctType>    (notice that I will drop the last template parameter)
//  Such object will contain all sub-objects for which 1) there are at least nVariables variables,
//                                                     2) the first variable has at most order maxOrder
//                                                     3) the total order is at most totalOrder
//  We split the basis functions with those characteristics into two groups:
//     the ones where the first variable has an order of exactly maxOrder
//     and the ones where the first variable has an order of maxOrder-1 or lower
//  Both those groups have to contain several basis fucntions and therfore coefficients. Therefore they will be objects containing further objects
//
//  The structure above is implementedby making the mentioned object contain two objects:
//     -> "top" of type
//            sparseFunctionStruct<totalOrdertop = totalOrder - maxOrder, maxOrdertop = totalOrder - maxOrder , nVariablestop = nVariables -1>
//                 Notice how this object will be in charge of handling one less variable. The remaining variables can have a total order which is
//                 the original one decreased by the order that the first variable has
//  and
//     -> "rest" of type
//            sparseFunctionStruct<totalOrderrest = totalOrder , maxOrderrest = maxOrder - 1, nVariablesrest = nVariables>
//                 This object instead will contain all the subobjects for which the first variable has at most order maxOrder - 1
//  tThis nesting structure ends when the order of the last variable is determined. In that case the innermost object stores the value of the coefficient.

#ifndef SparseFunctionStruct_hpp
#define SparseFunctionStruct_hpp

#include <vector>
#include <Eigen/Dense>
#include <cassert>
#include <iostream>
//#include <ranges>  // used for std::views::drop


namespace SparseFunction {

namespace Utilities {

// Utility Functions
inline void printOrderVectors(const std::vector<int>& orders, double coeff){
    std::cout << "[" << orders[0];
// Unfortunately std::views doesn't seem to work on MacOS  https://stackoverflow.com/questions/73628848/is-the-stdviews-namespace-not-available-in-xcodes-c?rq=1 the code below is kept for future reference
//        for (auto ord : orders | std::views::drop(1)){}
    for (int i=1; i<orders.size(); ++i){
        std::cout << "," << orders[i];
    }
    std::cout << "](" << coeff << ")\n";
}

} // namespace Utilities


namespace Internal {

// ====== sparseFunctionStruct =======
template <int totalOrder, int maxOrder, int nVar, template<int> typename basisType>
class sparseFunctionStruct;
    
// ======
template <template<int> typename basisType>
struct sparseFunctionStruct<0, 0, 1, basisType>{
//===================
    double coeff;     // c_[0,...] where ... is calling order list
//===================
//~~~~ Constants
    constexpr static int       numBasisFunction(int n){
        return n+1;
    }
//~~~~ Construction/Assignment
    void coeffVectorPtr(std::vector<double*>& pointerVector, int& counter){
        pointerVector[counter++] = &coeff;
    }
//~~~~ Evaluation
    template <typename Derived>
    auto                operator()(const Eigen::ArrayBase<Derived>& x){
        return basisType<0>::basisF(x.col(0)) * coeff;
    };
    template <typename Derived1, typename Derived2>
    static void         applyAllBasisFunctions(const Eigen::ArrayBase<Derived1>& x, Eigen::ArrayBase<Derived2>& result, int& counter){
        result.col(counter++) = basisType<0>::basisF(x.col(0)) ;
    }
    template <typename Derived1, typename Derived2, typename Derived3>
    static void         applyAllBasisFunctions(const Eigen::ArrayBase<Derived1>& x, Eigen::ArrayBase<Derived2>& result, int& counter, const Eigen::ArrayBase<Derived3>& restOfExpression){
        result.col(counter++) = basisType<0>::basisF(x.col(0)) * restOfExpression;
    }
//~~~~ I/O
    void                print(){
        std::cout << "f0[x1]";
    }
    std::vector<int>&   printCoeff(std::vector<int>& orders){
        orders[0] = 0;
        Utilities::printOrderVectors(orders, coeff);
        return orders;
    }

};

// ======
template <int totalOrder, int maxOrder, template<int> typename basisType>
struct sparseFunctionStruct<totalOrder, maxOrder, 1, basisType>{
    using RestType = sparseFunctionStruct<totalOrder-1, maxOrder-1, 1, basisType>;
//===================
    double coeff;      // c_[maxOrder, ...]
    RestType rest;
//===================
//~~~~ Constants
    constexpr static int       numBasisFunction(int n){
        return RestType::numBasisFunction(n)+1;
    }
//~~~~ Construction/Assignment
    void coeffVectorPtr(std::vector<double*>& pointerVector, int& counter){
        pointerVector[counter++] = &coeff;
        rest.coeffVectorPtr(pointerVector, counter);
    }
//~~~~ Evaluation
    template <typename Derived>
    auto                operator()(const Eigen::ArrayBase<Derived>& x){
        return basisType<maxOrder>::basisF(x.col(0)) * coeff + rest(x);
    };
    template <typename Derived1, typename Derived2>
    static void         applyAllBasisFunctions(const Eigen::ArrayBase<Derived1>& x, Eigen::ArrayBase<Derived2>& result, int& counter){
        result.col(counter++) = basisType<maxOrder>::basisF(x.col(0));
        RestType::applyAllBasisFunctions(x, result, counter);
    }
    template <typename Derived1, typename Derived2, typename Derived3>
    static void         applyAllBasisFunctions(const Eigen::ArrayBase<Derived1>& x, Eigen::ArrayBase<Derived2>& result, int& counter, const Eigen::ArrayBase<Derived3>& restOfExpression){
        result.col(counter++) = basisType<maxOrder>::basisF(x.col(0)) * restOfExpression;
        RestType::applyAllBasisFunctions(x, result, counter, restOfExpression);
    }
//~~~~ I/O
    void                print(){
        std::cout << "(f" << maxOrder << "[x1]";
        std::cout << "+";
        rest.print();
        std::cout << ")";
    }
    std::vector<int>&   printCoeff(std::vector<int>& orders){
        orders[0] = totalOrder;
        Utilities::printOrderVectors(orders, coeff);
        rest.printCoeff(orders);
        return orders;
    }
};

// ======
template <int totalOrder, int nVar, template<int> typename basisType>
struct sparseFunctionStruct<totalOrder, 0, nVar, basisType>{
    using TopType = sparseFunctionStruct<totalOrder, totalOrder , nVar-1, basisType>;
//===================
    TopType top;
//===================
//~~~~ Constants
    constexpr static int       numBasisFunction(int n){
        return TopType::numBasisFunction(n);
    }
//~~~~ Construction/Assignment
    void coeffVectorPtr(std::vector<double*>& pointerVector, int& counter){
        top.coeffVectorPtr(pointerVector, counter);
    }
//~~~~ Evaluation
    template <typename Derived>
    auto                operator()(const Eigen::ArrayBase<Derived>& x){
        return basisType<0>::basisF(x.col(nVar-1)) * top(x);
    };
    template <typename Derived1, typename Derived2>
    static void         applyAllBasisFunctions(const Eigen::ArrayBase<Derived1>& x, Eigen::ArrayBase<Derived2>& result, int& counter){
        TopType::applyAllBasisFunctions(x, result, counter, basisType<0>::basisF(x.col(nVar-1)));
    }
    template <typename Derived1, typename Derived2, typename Derived3>
    static void         applyAllBasisFunctions(const Eigen::ArrayBase<Derived1>& x, Eigen::ArrayBase<Derived2>& result, int& counter, const Eigen::ArrayBase<Derived3>& restOfExpression){
        TopType::applyAllBasisFunctions(x, result, counter, basisType<0>::basisF(x.col(nVar-1)) * restOfExpression);
    }
//~~~~ I/O
    void                print(){
        std::cout << "(f0[x" << nVar << "]";
        top.print();
        std::cout << ")";
    }
    std::vector<int>&   printCoeff(std::vector<int>& orders){
        orders[nVar-1] = 0;
        top.printCoeff(orders);
        return orders;
    }
    
};

// ======
template <int totalOrder, int maxOrder, int nVar, template<int> typename basisType>
struct sparseFunctionStruct {
    using TopType = sparseFunctionStruct<totalOrder - maxOrder, totalOrder - maxOrder, nVar-1, basisType>;
    using RestType = sparseFunctionStruct<totalOrder, maxOrder-1, nVar, basisType>;
//===================
    TopType top;
    RestType rest;
//===================
//~~~~ Constants
    constexpr static int       numBasisFunction(int n){
        return TopType::numBasisFunction(RestType::numBasisFunction(n));
    }
//~~~~ Construction/Assignment
    void coeffVectorPtr(std::vector<double*>& pointerVector, int& counter){
        top.coeffVectorPtr(pointerVector,counter);
        rest.coeffVectorPtr(pointerVector,counter);
    }
//~~~~ Evaluation
    template <typename Derived>
    auto                operator()(const Eigen::ArrayBase<Derived>& x){
        return basisType<maxOrder>::basisF(x.col(nVar-1)) * top(x) + rest(x);
    };
    template <typename Derived1, typename Derived2>
    static void         applyAllBasisFunctions(const Eigen::ArrayBase<Derived1>& x, Eigen::ArrayBase<Derived2>& result, int& counter){
        TopType::applyAllBasisFunctions(x, result, counter, basisType<maxOrder>::basisF(x.col(nVar-1)));
        RestType::applyAllBasisFunctions(x, result, counter);
    }
    template <typename Derived1, typename Derived2, typename Derived3>
    static void         applyAllBasisFunctions(const Eigen::ArrayBase<Derived1>& x, Eigen::ArrayBase<Derived2>& result, int& counter, const Eigen::ArrayBase<Derived3>& restOfExpression){
        TopType::applyAllBasisFunctions(x, result, counter, basisType<maxOrder>::basisF(x.col(nVar-1)) * restOfExpression);
        RestType::applyAllBasisFunctions(x, result, counter, restOfExpression);
    }
//~~~~ I/O
    void                print(){
        std::cout << "(f" << maxOrder << "[x" << nVar << "]*";
        top.print();
        std::cout << "+";
        rest.print();
        std::cout << ")";
    }
    std::vector<int>&   printCoeff(std::vector<int>& orders){
        orders[nVar-1] = maxOrder;
        top.printCoeff(orders);
        rest.printCoeff(orders);
        return orders;
    }
};

} // namespace Internal

} // namespace SparseFunction

#endif /* SparseFunctionStruct_hpp */
