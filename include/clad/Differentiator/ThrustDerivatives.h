#ifndef CLAD_DIFFERENTIATOR_THRUSTDERIVATIVES_H
#define CLAD_DIFFERENTIATOR_THRUSTDERIVATIVES_H

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#include "clad/Differentiator/Differentiator.h"

// A simple host-side wrapper for the thrust::reduce call
inline double reduce_sum_of_doubles(const double* first, const double* last) {
    return thrust::reduce(first, last, 0.0, thrust::plus<double>());
}

namespace clad {
namespace custom_derivatives {

// Custom pullback for wrapper function
__device__ void reduce_sum_of_doubles_pullback(
    const double* first, const double* last,
    const double& d_output, 
    double* d_first,        
    double* d_last          
) {
    size_t size = last - first;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < size;
         i += blockDim.x * gridDim.x)
    {
        d_first[i] += d_output;
    }
}

} // namespace custom_derivatives
} // namespace clad

#endif // CLAD_DIFFERENTIATOR_THRUSTDERIVATIVES_H