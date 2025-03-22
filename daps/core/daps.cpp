// daps.cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <functional>

// Define pi
const double PI = 3.141592653589793238463;

// Define the result struct
struct DAPSResult {
    std::vector<double> x;
    double fun_val;
    int nfev;
    int nit;
    bool success;
    int final_prime_idx_x;
    int final_prime_idx_y;
    int final_prime_idx_z;
    int dimensions;
};

// External Python callback function declaration
extern "C" bool call_python_callback(const std::vector<double>& x, double fun_val, int evals, void* py_callback_ptr);

// Factory function that creates a wrapper around a Python function for 1D
inline std::function<double(double, double, double)> create_1d_py_func_wrapper(void* py_func_ptr) {
    return [py_func_ptr](double x, double /* unused y */, double /* unused z */) -> double {
        std::vector<double> coords = {x, 0.0, 0.0}; // Only x is used
        double result = 0.0;
        
        // Call into Python with evals = -1 to indicate 1D function evaluation
        // The result will be stored in the first element of coords
        call_python_callback(coords, 0.0, -1, py_func_ptr);
        
        // Return the result from the first element (set by Python callback)
        result = coords[0];
        return result;
    };
}

// Factory function that creates a wrapper around a Python function for 2D
inline std::function<double(double, double, double)> create_2d_py_func_wrapper(void* py_func_ptr) {
    return [py_func_ptr](double x, double y, double /* unused z */) -> double {
        std::vector<double> coords = {x, y, 0.0}; // Only x and y are used
        double result = 0.0;
        
        // Call into Python with evals = -2 to indicate 2D function evaluation
        // The result will be stored in the first element of coords
        call_python_callback(coords, 0.0, -2, py_func_ptr);
        
        // Return the result from the first element (set by Python callback)
        result = coords[0];
        return result;
    };
}

// Factory function that creates a wrapper around a Python function for 3D
inline std::function<double(double, double, double)> create_3d_py_func_wrapper(void* py_func_ptr) {
    return [py_func_ptr](double x, double y, double z) -> double {
        std::vector<double> coords = {x, y, z}; // All three coordinates are used
        double result = 0.0;
        
        // Call into Python with evals = -3 to indicate 3D function evaluation
        // The result will be stored in the first element of coords
        call_python_callback(coords, 0.0, -3, py_func_ptr);
        
        // Return the result from the first element (set by Python callback)
        result = coords[0];
        return result;
    };
}

// Simple prime number generator (for small primes)
inline int get_prime(int n) {
    // Pre-defined list of prime numbers
    static const int primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};
    static const int num_primes = sizeof(primes) / sizeof(primes[0]);
    
    // Handle bounds
    if (n < 0) return primes[0];
    if (n >= num_primes) return primes[num_primes - 1];
    
    return primes[n];
}

// Sieve of Eratosthenes - Generate prime numbers up to a limit
inline std::vector<int> generate_primes(int n) {
    std::vector<int> primes;
    
    for (int i = 0; i < n; i++) {
        primes.push_back(get_prime(i));
    }
    
    return primes;
}

// Recursive Fractal Cliff Valley function
inline double recursive_fractal_cliff_valley(double x, double y, double z) {
    // Local minima creation
    double local_pattern = std::sin(x * y * z) * std::cos(x + y - z) + 
                           std::sin(x * PI) * std::sin(y * PI) * std::sin(z * PI);
    
    // Fractal recursion (scaled down)
    double fractal_component = 0.0;
    if (std::abs(x) < 2.0 && std::abs(y) < 2.0 && std::abs(z) < 2.0) {
        fractal_component = 0.2 * recursive_fractal_cliff_valley(x * 2.0, y * 2.0, z * 2.0);
    }
    
    // Valley structure
    double valley = std::pow(x - PI, 2) + std::pow(y - std::exp(1.0), 2) + std::pow(z - std::sqrt(5.0), 2);
    
    // Cliff structures
    double cliff = 5.0 / (1.0 + std::exp(-5.0 * (std::sin(x * y) + std::sin(y * z) + std::sin(z * x))));
    
    // Combine components
    return valley + local_pattern + cliff + fractal_component;
}

// Generic DAPS Algorithm for 1D, 2D, and 3D
inline DAPSResult daps_optimize(
    std::function<double(double, double, double)> func, 
    double x_min, double x_max, 
    double y_min, double y_max, 
    double z_min, double z_max,
    int max_iter = 10000,
    int min_prime_idx_x = 0,
    int min_prime_idx_y = 0,
    int min_prime_idx_z = 0,
    void* callback_ptr = nullptr,
    double tol = 1e-8,
    int dimensions = 3
) {
    // Validate dimensions
    if (dimensions < 1 || dimensions > 3) {
        throw std::invalid_argument("Dimensions must be 1, 2, or 3");
    }
    
    // Generate primes
    std::vector<int> primes = generate_primes(1000);
    
    // Ensure valid prime indices
    min_prime_idx_x = std::max(0, min_prime_idx_x);
    min_prime_idx_y = std::max(0, min_prime_idx_y);
    min_prime_idx_z = std::max(0, min_prime_idx_z);
    int max_prime_idx = static_cast<int>(primes.size()) - 1;
    
    // Initialize variables
    std::vector<double> best_solution(dimensions, 0.0);
    double best_value = std::numeric_limits<double>::infinity();
    int evaluations = 0;
    int iter = 0;
    bool success = false;
    
    // Current prime indices
    int p_idx_x = min_prime_idx_x;
    int p_idx_y = (dimensions >= 2) ? min_prime_idx_y : 0;
    int p_idx_z = (dimensions >= 3) ? min_prime_idx_z : 0;
    
    // Use the y and z values if in lower dimensions
    double fixed_y = (dimensions < 2) ? (y_min + y_max) / 2.0 : 0.0;
    double fixed_z = (dimensions < 3) ? (z_min + z_max) / 2.0 : 0.0;
    
    // Adaptive search
    while (iter < max_iter) {
        // Calculate step sizes for each dimension
        double x_step = (x_max - x_min) / primes[p_idx_x];
        double y_step = (dimensions >= 2) ? (y_max - y_min) / primes[p_idx_y] : 0.0;
        double z_step = (dimensions >= 3) ? (z_max - z_min) / primes[p_idx_z] : 0.0;
        
        // Flag to check if we found a better solution
        bool improved = false;
        
        // Search using the prime-based grid
        for (int i = 0; i < primes[p_idx_x]; i++) {
            double x = x_min + i * x_step;
            
            // If 1D, evaluate with fixed y and z
            if (dimensions == 1) {
                double value = func(x, fixed_y, fixed_z);
                evaluations++;
                
                if (value < best_value) {
                    best_solution[0] = x;
                    best_value = value;
                    improved = true;
                    
                    // Call callback if provided
                    if (callback_ptr) {
                        std::vector<double> callback_x = {x};
                        if (!call_python_callback(callback_x, best_value, evaluations, callback_ptr)) {
                            // Early stopping
                            return {callback_x, best_value, evaluations, iter + 1, false, p_idx_x, p_idx_y, p_idx_z, dimensions};
                        }
                    }
                    
                    // Check if we've reached the tolerance
                    if (best_value <= tol) {
                        success = true;
                        return {{x}, best_value, evaluations, iter + 1, success, p_idx_x, p_idx_y, p_idx_z, dimensions};
                    }
                }
                continue;  // Skip to next x value
            }
            
            // 2D or 3D case
            for (int j = 0; j < ((dimensions >= 2) ? primes[p_idx_y] : 1); j++) {
                double y = (dimensions >= 2) ? (y_min + j * y_step) : fixed_y;
                
                // If 2D, evaluate with fixed z
                if (dimensions == 2) {
                    double value = func(x, y, fixed_z);
                    evaluations++;
                    
                    if (value < best_value) {
                        best_solution[0] = x;
                        best_solution[1] = y;
                        best_value = value;
                        improved = true;
                        
                        // Call callback if provided
                        if (callback_ptr) {
                            std::vector<double> callback_x = {x, y};
                            if (!call_python_callback(callback_x, best_value, evaluations, callback_ptr)) {
                                // Early stopping
                                return {callback_x, best_value, evaluations, iter + 1, false, p_idx_x, p_idx_y, p_idx_z, dimensions};
                            }
                        }
                        
                        // Check if we've reached the tolerance
                        if (best_value <= tol) {
                            success = true;
                            return {{x, y}, best_value, evaluations, iter + 1, success, p_idx_x, p_idx_y, p_idx_z, dimensions};
                        }
                    }
                    continue;  // Skip to next y value
                }
                
                // 3D case
                for (int k = 0; k < ((dimensions >= 3) ? primes[p_idx_z] : 1); k++) {
                    double z = (dimensions >= 3) ? (z_min + k * z_step) : fixed_z;
                    
                    // Evaluate the function
                    double value = func(x, y, z);
                    evaluations++;
                    
                    // Check if better solution
                    if (value < best_value) {
                        best_solution[0] = x;
                        if (dimensions >= 2) best_solution[1] = y;
                        if (dimensions >= 3) best_solution[2] = z;
                        best_value = value;
                        improved = true;
                        
                        // Call callback if provided
                        if (callback_ptr) {
                            std::vector<double> callback_x = {x, y, z};
                            callback_x.resize(dimensions);  // Ensure correct size
                            if (!call_python_callback(callback_x, best_value, evaluations, callback_ptr)) {
                                // Early stopping
                                return {best_solution, best_value, evaluations, iter + 1, false, p_idx_x, p_idx_y, p_idx_z, dimensions};
                            }
                        }
                        
                        // Check if we've reached the tolerance
                        if (best_value <= tol) {
                            success = true;
                            return {best_solution, best_value, evaluations, iter + 1, success, p_idx_x, p_idx_y, p_idx_z, dimensions};
                        }
                    }
                }
            }
        }
        
        // Refine the search region
        if (improved) {
            // Narrow the search region around the best solution
            double half_x_step = x_step * 2.0;
            double half_y_step = (dimensions >= 2) ? y_step * 2.0 : 0.0;
            double half_z_step = (dimensions >= 3) ? z_step * 2.0 : 0.0;
            
            // Update bounds ensuring they stay within global bounds
            x_min = std::max(x_min, best_solution[0] - half_x_step);
            x_max = std::min(x_max, best_solution[0] + half_x_step);
            
            if (dimensions >= 2) {
                y_min = std::max(y_min, best_solution[1] - half_y_step);
                y_max = std::min(y_max, best_solution[1] + half_y_step);
            }
            
            if (dimensions >= 3) {
                z_min = std::max(z_min, best_solution[2] - half_z_step);
                z_max = std::min(z_max, best_solution[2] + half_z_step);
            }
            
            // Increase prime indices to get more refinement
            p_idx_x = std::min(p_idx_x + 1, max_prime_idx);
            if (dimensions >= 2) p_idx_y = std::min(p_idx_y + 1, max_prime_idx);
            if (dimensions >= 3) p_idx_z = std::min(p_idx_z + 1, max_prime_idx);
        } else {
            // If no improvement, try different prime indices
            p_idx_x = (p_idx_x + 1) % (max_prime_idx - min_prime_idx_x + 1) + min_prime_idx_x;
            if (dimensions >= 2) p_idx_y = (p_idx_y + 1) % (max_prime_idx - min_prime_idx_y + 1) + min_prime_idx_y;
            if (dimensions >= 3) p_idx_z = (p_idx_z + 1) % (max_prime_idx - min_prime_idx_z + 1) + min_prime_idx_z;
        }
        
        iter++;
    }
    
    return {best_solution, best_value, evaluations, iter, success, p_idx_x, p_idx_y, p_idx_z, dimensions};
}

// Rosenbrock 3D function
inline double rosenbrock_3d(double x, double y, double z) {
    return 100.0 * std::pow(y - x * x, 2) + std::pow(x - 1.0, 2) +
           100.0 * std::pow(z - y * y, 2) + std::pow(y - 1.0, 2);
}

// Sphere function
inline double sphere_function(double x, double y, double z) {
    return x * x + y * y + z * z;
}

// Ackley function
inline double ackley_function(double x, double y, double z) {
    double a = 20.0;
    double b = 0.2;
    double c = 2.0 * PI;
    
    double sum_sq = x * x + y * y + z * z;
    double sum_cos = std::cos(c * x) + std::cos(c * y) + std::cos(c * z);
    
    return -a * std::exp(-b * std::sqrt(sum_sq / 3.0)) - std::exp(sum_cos / 3.0) + a + std::exp(1.0);
}

// Rastrigin function
inline double rastrigin_function(double x, double y, double z) {
    return 30.0 + (x * x - 10.0 * std::cos(2.0 * PI * x)) +
                 (y * y - 10.0 * std::cos(2.0 * PI * y)) +
                 (z * z - 10.0 * std::cos(2.0 * PI * z));
} 
