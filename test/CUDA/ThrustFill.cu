// RUN: %cladclang_cuda -I%S/../../include --cuda-path=%cudapath \
// RUN:     --cuda-gpu-arch=%cudaarch %cudaldflags -oThrustFill.out \
// RUN:     -Xclang -verify %s 2>&1 | %filecheck %s
//
// RUN: ./ThrustFill.out | %filecheck_exec %s
//
// REQUIRES: cuda-runtime
//
// expected-no-diagnostics

#include <iostream>
#include <vector>
#include <iomanip>

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/ThrustDerivatives.h"
#include "../TestUtils.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

void fill_array(thrust::device_vector<double>& vec, double value) {
    thrust::fill(vec.begin(), vec.end(), value);
}
// CHECK: void fill_array_pullback(thrust::device_vector<double> &vec, double value, thrust::device_vector<double> *_d_vec, double *_d_value) {
// CHECK-NEXT: {
// CHECK-NEXT:     iterator _r0 = std::begin((*_d_vec));
// CHECK-NEXT:     iterator _r1 = std::end((*_d_vec));
// CHECK-NEXT:     clad::custom_derivatives::thrust::fill_pullback(std::begin(vec), std::end(vec), value, &_r0, &_r1, _d_value);
// CHECK-NEXT: }
// CHECK-NEXT: }

int main() {
    INIT_GRADIENT(fill_array);

    std::vector<double> host_input = {10.0, 5.0, 2.0, 20.0};
    thrust::device_vector<double> device_input = host_input;

    double value_to_fill = 5.0;
    double d_value = 0.0;
    
    thrust::device_vector<double> d_vec(host_input.size());
    std::vector<double> d_vec_host = {1.0, 2.0, 3.0, 4.0};
    d_vec = d_vec_host;

    fill_array_grad.execute(device_input, value_to_fill, &d_vec, &d_value);

    thrust::host_vector<double> host_d_vec = d_vec;
    printf("d_vec: %.3f %.3f %.3f %.3f\n", host_d_vec[0], host_d_vec[1], host_d_vec[2], host_d_vec[3]);
    // CHECK-EXEC: d_vec: 0.000 0.000 0.000 0.000

    printf("d_value: %.3f\n", d_value);
    // CHECK-EXEC: d_value: 10.000

    return 0;
}
