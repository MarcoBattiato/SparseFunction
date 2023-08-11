//
//  main.cpp
//  SparseFunctions
//
//  Created by Marco Battiato on 3/5/23.
//
//  This is a work in progress to allow for the creation basis sets for the approximation of multi-dimensional functions.
//  The multidimensional basis set is built from a 1-dim basis set characterised by an order. However instead of including
//  all the combinations of basis set obtained by a cartesian product (with a number of basis function of ~N^D),
//  only D-dim basis functions up to a given total order N (sum of all the orders in each direction) are included
//  (leading to an asymptotic (d->inf) number of basis function of ~(1/Sqrt[D Pi]) (2 Pi e/d)^(D/2) (N/2)^D)
//
//  So far only monomials are implemented as 1-D basis functions

#include "SparseFunction.hpp"
#include "BasisFunctionsSets.hpp"
#include <iostream>
#include <chrono>

using std::cout;
using namespace SparseFunction;

int main(int argc, const char * argv[]) {
    
    const int order = 5, numVar = 6;
    sparseFunction<order, numVar, TaylorBasis> bla;
    // bla will be a polynomial in 3 variables of maximum degree 4; TaylorBasis means simple monomials

    // Returns the number of monomials, or in other terms the number of coefficients of the polynomial, or in other terms the number of basis functions
    cout << "N basis functions " << bla.numBasisFunction() << "\n\n";

    // returns a vector of pointers at the coefficients. Notice how the addresses are all one word apart, meaning that the compiler has managed to
    // optimise the recursive objects very well
    auto coeffPoint = bla.coeffVectorPtr();
    for (auto p : coeffPoint) {cout << p << " ";}
    cout << "\n\n";
    // The poiters to the actual data are not to be used, I made that function just to test the performance
    
    // I test here the direct assignment of the value of the coefficients by accessing the memory directly
    // Obviously the user should not do it in this way
    for (auto p : coeffPoint) {*p = 0;}
    
    // This function prints the value of the coefficients after a vector of the order of the assiciated monomial
    bla.printCoeff(); cout << "\n\n";
    cout << "\n";

    
    // I now generate some data. I will use the data to then find the polynomial that best fits the data
    const int nDataPoints = 500;
    Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic> v = Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic>::Random(nDataPoints, numVar);
    // The instruction above builds some random points in numVar dimensions. They are random because I wanted to test how good is the routing in fitting not on a regulat grid
    
    Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic> y = 3.1*v.col(0) + 2.*v.col(1) + 3.*v.col(2)*v.col(1) + 1.2;
    // I now construct the numerical values at the points above. I build the values by evaluating the polynomial 3.1 x + 2 y + 3 y z + 1.2
    // If the fitting works well the fitted coefficients should be very close to the ones above
    
    // bla.fitData(v, y) calculates the coefficient of bla such that it approximates the passed points the best.
    // The return value is the fitting error. with the values above the fitting error should be very small
    // If instead of using a polynomial expression, we had used a more generic function to construct y, the error would be finite, since there is no way
    // a polynomial can approximate exactly a generic function
    cout << bla.fitData(v, y)<< "\n\n";
    
    // Now check that the coefficients from the fitting are sufficiently close to the ones used to define y
    bla.printCoeff();
    cout << "\n\n";
    
    // Let us now see how to evaluate the polynomial
    cout << bla(v).transpose() << "\n";
    cout << y.transpose() << "\n\n";
    
    return 0;
}


//    Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic>::Random(nDataPoints, 1);
   
//    Eigen::Array<double,nDataPoints,1> sol;
//    auto start = std::chrono::high_resolution_clock::now();
//    sol = bla(v) ;
//    auto finish = std::chrono::high_resolution_clock::now();
//    cout << std::chrono::duration<double>(finish - start).count()<< "\n\n";

//    v = Eigen::Array<double,Eigen::Dynamic,numVar>::Random(nDataPoints, numVar);
//    start = std::chrono::high_resolution_clock::now();
//    sol = bla(v);
//    finish = std::chrono::high_resolution_clock::now();
//    cout << std::chrono::duration<double>(finish - start).count()<< "\n\n";
    
//    v = Eigen::Array<double,Eigen::Dynamic,numVar>::Random(nDataPoints, numVar);
//    start = std::chrono::high_resolution_clock::now();
//    sol = bla(v);
//    finish = std::chrono::high_resolution_clock::now();
//    cout << std::chrono::duration<double>(finish - start).count()<< "\n\n";
 
    
//    Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic> sol2;
//    sol2.resize(nDataPoints, bla.numBasisFunction());
//    start = std::chrono::high_resolution_clock::now();
//    sol2 = bla.applyAllBasisFunctionsVerySlow(v) ;
//    finish = std::chrono::high_resolution_clock::now();
//    cout << std::chrono::duration<double>(finish - start).count()<< "\n\n";
//
//
//    Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic> sol3;
//    sol3.resize(nDataPoints, bla.numBasisFunction());
//    start = std::chrono::high_resolution_clock::now();
//    sol3 = bla.applyAllBasisFunctions(v) ;
//    finish = std::chrono::high_resolution_clock::now();
//    cout << std::chrono::duration<double>(finish - start).count()<< "\n\n";

//    cout << v << "\n\n";
//    cout << sol2 << "\n\n";
//    cout << sol3 << "\n\n";
//    myFun(sol2);
    
//    cout << bla.fitData(v, y)<< "\n\n";
//
//    bla.printCoeff();
    
//    const int size = 10000;
//    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> aMat = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>::Random(size, size);
//
//    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> bVec = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>::Random(size, 1);
//
//    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> sol = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>::Random(size, 1);
//
//
//    auto start = std::chrono::high_resolution_clock::now();
//    sol = aMat.householderQr().solve(bVec) ;
//    auto finish = std::chrono::high_resolution_clock::now();
//    cout <<sol.transpose() << "\n\n";
//    cout << std::chrono::duration<double>(finish - start).count()<< "\n\n";
