#include <iostream>
#include <vector>

// Thrust headers
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/memory.h>

// Clad headers
#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/ThrustDerivatives.h"

// The function we want to differentiate.
double sum_array(double* data, size_t size) {
    return thrust_reduce(data, data + size, 0.0, thrust::plus<double>());
}

int main() {
    std::vector<double> host_input = {15.0, 25.0, 35.0, 45.0, 55.0};
    std::cout << "Original host vector: ";
    for (const auto& val : host_input) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    thrust::device_vector<double> device_input = host_input;

    auto sum_grad_func = clad::gradient(sum_array, "data");

    thrust::device_vector<double> device_gradients(host_input.size(), 0.0);

    double* p_device_input = thrust::raw_pointer_cast(device_input.data());
    double* p_device_gradients = thrust::raw_pointer_cast(device_gradients.data());

    sum_grad_func.execute(p_device_input, host_input.size(), p_device_gradients);

    thrust::host_vector<double> host_gradients = device_gradients;

    std::cout << "Computed gradients:   ";
    for (size_t i = 0; i < host_gradients.size(); ++i) {
        std::cout << host_gradients[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
// clang-20: /home/elrawy/gsoc/clad/lib/Differentiator/CladUtils.cpp:240: DeclContext *clad::utils::FindDeclContext(clang::Sema &, clang::DeclContext *, const clang::DeclContext *): Assertion `isa<NamespaceDecl>(DC2) && "DC2 should only consists of namespace, CXXRecord and " "translation unit declaration context."' failed.