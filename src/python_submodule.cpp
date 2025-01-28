#include "python_submodule.h"

// Python Binding and Time_Quad class instances.
PYBIND11_MODULE(time_quadratures, m) {
    m.doc() = "Fast multithreaded caculations of cyclical autocorrelations .\n";
    init_autocorr(m);
}
