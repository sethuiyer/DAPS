// daps.cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <functional>
#include <tuple>

// Define pi
const double PI = 3.141592653589793238463;

// Simple prime number generator (for small primes)
int get_prime(int n) {
    int primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233}; // Add more as needed
    return primes[n - 1]; // Adjust index (primes start at 2, index 0)
}

// Sieve of Eratosthenes - Generate prime numbers up to a limit
std::vector<int> generate_primes(int limit) {
    std::vector<bool> sieve(limit + 1, true);
    std::vector<int> primes;
    
    for (int p = 2; p * p <= limit; p++) {
        if (sieve[p]) {
            for (int i = p * p; i <= limit; i += p) {
                sieve[i] = false;
            }
        }
    }
    
    for (int p = 2; p <= limit; p++) {
        if (sieve[p]) {
            primes.push_back(p);
        }
    }
    
    return primes;
}

// Recursive Fractal Cliff Valley function
double recursive_fractal_cliff_valley(double x, double y, double z) {
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

// DAPS Algorithm
std::tuple<std::vector<double>, double, int, int, bool, int, int> daps_optimize(
    std::function<double(double, double, double)> func, 
    double x_min, double x_max, 
    double y_min, double y_max, 
    double z_min, double z_max,
    int max_iter = 10000,
    int min_prime_idx = 0,
    int max_prime_idx = 30,
    std::function<bool(const std::vector<double>&, double, int)> callback = nullptr,
    double tol = 1e-8
) {
    // Generate primes
    std::vector<int> primes = generate_primes(1000);
    
    // Ensure valid prime indices
    min_prime_idx = std::max(0, min_prime_idx);
    max_prime_idx = std::min(static_cast<int>(primes.size()) - 1, max_prime_idx);
    
    // Initialize variables
    std::vector<double> best_solution = {0, 0, 0};
    double best_value = std::numeric_limits<double>::infinity();
    int evaluations = 0;
    int iter = 0;
    bool success = false;
    
    // Current prime indices
    int p_idx_x = min_prime_idx;
    int p_idx_y = min_prime_idx;
    int p_idx_z = min_prime_idx;
    
    // Adaptive search in 3D
    while (iter < max_iter) {
        // Calculate step sizes for each dimension
        double x_step = (x_max - x_min) / primes[p_idx_x];
        double y_step = (y_max - y_min) / primes[p_idx_y];
        double z_step = (z_max - z_min) / primes[p_idx_z];
        
        // Flag to check if we found a better solution
        bool improved = false;
        
        // Search using the prime-based grid
        for (int i = 0; i < primes[p_idx_x]; i++) {
            double x = x_min + i * x_step;
            for (int j = 0; j < primes[p_idx_y]; j++) {
                double y = y_min + j * y_step;
                for (int k = 0; k < primes[p_idx_z]; k++) {
                    double z = z_min + k * z_step;
                    
                    // Evaluate the function
                    double value = func(x, y, z);
                    evaluations++;
                    
                    // Check if better solution
                    if (value < best_value) {
                        best_solution = {x, y, z};
                        best_value = value;
                        improved = true;
                        
                        // Call callback if provided
                        if (callback) {
                            if (!callback(best_solution, best_value, evaluations)) {
                                // Early stopping
                                return {best_solution, best_value, evaluations, iter + 1, false, p_idx_x, p_idx_y};
                            }
                        }
                        
                        // Check if we've reached the tolerance
                        if (best_value <= tol) {
                            success = true;
                            return {best_solution, best_value, evaluations, iter + 1, success, p_idx_x, p_idx_y};
                        }
                    }
                }
            }
        }
        
        // Refine the search region
        if (improved) {
            // Narrow the search region around the best solution
            double half_x_step = x_step * 2.0;
            double half_y_step = y_step * 2.0;
            double half_z_step = z_step * 2.0;
            
            // Update bounds ensuring they stay within global bounds
            x_min = std::max(x_min, best_solution[0] - half_x_step);
            x_max = std::min(x_max, best_solution[0] + half_x_step);
            y_min = std::max(y_min, best_solution[1] - half_y_step);
            y_max = std::min(y_max, best_solution[1] + half_y_step);
            z_min = std::max(z_min, best_solution[2] - half_z_step);
            z_max = std::min(z_max, best_solution[2] + half_z_step);
            
            // Increase prime indices to get more refinement
            p_idx_x = std::min(p_idx_x + 1, max_prime_idx);
            p_idx_y = std::min(p_idx_y + 1, max_prime_idx);
            p_idx_z = std::min(p_idx_z + 1, max_prime_idx);
        } else {
            // If no improvement, try different prime indices
            p_idx_x = (p_idx_x + 1) % (max_prime_idx - min_prime_idx + 1) + min_prime_idx;
            p_idx_y = (p_idx_y + 1) % (max_prime_idx - min_prime_idx + 1) + min_prime_idx;
            p_idx_z = (p_idx_z + 1) % (max_prime_idx - min_prime_idx + 1) + min_prime_idx;
        }
        
        iter++;
    }
    
    return {best_solution, best_value, evaluations, iter, success, p_idx_x, p_idx_y};
}

// Rosenbrock 3D function
double rosenbrock_3d(double x, double y, double z) {
    return 100.0 * std::pow(y - x * x, 2) + std::pow(x - 1.0, 2) +
           100.0 * std::pow(z - y * y, 2) + std::pow(y - 1.0, 2);
}

// Sphere function
double sphere_function(double x, double y, double z) {
    return x * x + y * y + z * z;
}

// Ackley function
double ackley_function(double x, double y, double z) {
    double a = 20.0;
    double b = 0.2;
    double c = 2.0 * PI;
    
    double sum_sq = x * x + y * y + z * z;
    double sum_cos = std::cos(c * x) + std::cos(c * y) + std::cos(c * z);
    
    return -a * std::exp(-b * std::sqrt(sum_sq / 3.0)) - std::exp(sum_cos / 3.0) + a + std::exp(1.0);
}

// Rastrigin function
double rastrigin_function(double x, double y, double z) {
    return 30.0 + (x * x - 10.0 * std::cos(2.0 * PI * x)) +
                 (y * y - 10.0 * std::cos(2.0 * PI * y)) +
                 (z * z - 10.0 * std::cos(2.0 * PI * z));
} 