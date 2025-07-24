#ifndef CLAD_DIFFERENTIATOR_THRUSTDERIVATIVES_H
#define CLAD_DIFFERENTIATOR_THRUSTDERIVATIVES_H

#include <iterator>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

#include "clad/Differentiator/Differentiator.h"

// Wrapper function for thrust::reduce
template <typename Iterator, typename T, typename BinaryOp>
T thrust_reduce(Iterator first, Iterator last, T init, BinaryOp op) {
    return thrust::reduce(first, last, init, op);
}

namespace clad {
namespace custom_derivatives {

// Custom pullback for our thrust_reduce wrapper function
template <typename Iterator, typename T, typename BinaryOp>
void thrust_reduce_pullback(
    Iterator first, Iterator last, T init, BinaryOp op,
    const T& d_output, 
    Iterator d_first,  
    Iterator d_last    
) {
    thrust::transform(
        d_first,                        
        d_last,                         
        d_first,                        
        thrust::placeholders::_1 + d_output 
    );
}

} // namespace custom_derivatives
} // namespace clad

#endif // CLAD_DIFFERENTIATOR_THRUSTDERIVATIVES_H