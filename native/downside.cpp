#include <cmath>
#include <cstddef>
#include <limits>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

// native.py supplies a valid pointer to count finite doubles.
extern "C" EXPORT double downside_deviation(const double* values, std::size_t count) {
    if (values == nullptr || count == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double total = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] < 0.0) {
            total += values[i] * values[i];
        }
    }
    return std::sqrt(total / static_cast<double>(count));
}
