
# _daps.pyx
# distutils: language = c++
# cython: language_level=3

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.functional cimport function
from libcpp cimport bool as cpp_bool
from cpython.object cimport PyObject_IsTrue
from cpython.ref cimport PyObject

# Registry for Python callbacks
cdef dict _callback_registry = {}

# -------------------------------------------------------------
# Extern declarations for C++ functions/structs
# -------------------------------------------------------------
cdef extern from "daps.cpp":
    double recursive_fractal_cliff_valley(double x, double y, double z) nogil
    double rosenbrock_3d(double x, double y, double z) nogil
    double sphere_function(double x, double y, double z) nogil
    double ackley_function(double x, double y, double z) nogil
    double rastrigin_function(double x, double y, double z) nogil

    function[double(double,double,double)] create_1d_py_func_wrapper(void* func_ptr) nogil
    function[double(double,double,double)] create_2d_py_func_wrapper(void* func_ptr) nogil
    function[double(double,double,double)] create_3d_py_func_wrapper(void* func_ptr) nogil

    cdef struct DAPSResult:
        vector[double] x
        double fun_val
        int nfev
        int nit
        cpp_bool success
        int final_prime_idx_x
        int final_prime_idx_y
        int final_prime_idx_z
        int dimensions

    DAPSResult daps_optimize(
        function[double(double,double,double)] func,
        double x_min, double x_max,
        double y_min, double y_max,
        double z_min, double z_max,
        int max_iter,
        int min_prime_idx_x,
        int min_prime_idx_y,
        int min_prime_idx_z,
        void* callback_ptr,
        double tol,
        int dimensions
    ) nogil

# -------------------------------------------------------------
# Extern-C wrapper to match C linkage for call_python_callback
# -------------------------------------------------------------
    cdef public extern "C" bint call_python_callback(
    const vector[double]& x,
    double fun_val,
    int evals,
    void* ptr
    ) with gil:
        cdef long cid = <long>ptr
        cdef bint keep_going = True
        try:
            pyf = _callback_registry.get(cid)
            if pyf is None:
                return keep_going
            x_list = [x[i] for i in range(x.size())]
            py_res = pyf(x_list, fun_val, evals)
            if py_res is not None:
                keep_going = <bint>PyObject_IsTrue(py_res)
        except:
            keep_going = True
        return keep_going


# Cython-implemented callback (with gil)
cdef cpp_bool _cy_call_python_callback(const vector[double]& x, double fun_val, int evals, void* ptr) with gil:
    cdef long cid = <long>ptr
    cdef cpp_bool keep = True
    try:
        pyf = _callback_registry.get(cid)
        if pyf is None:
            return keep
        x_list = [x[i] for i in range(x.size())]
        res = pyf(x_list, fun_val, evals)
        if res is not None:
            keep = <cpp_bool>PyObject_IsTrue(res)
    except Exception:
        keep = True
    return keep

# -------------------------------------------------------------
# Wrapper factory for Python objectives
# -------------------------------------------------------------
cdef function[double(double,double,double)] create_function_wrapper(object py_func, int dims):
    cdef long fid = id(py_func)
    _callback_registry[fid] = py_func
    if dims == 1:
        return create_1d_py_func_wrapper(<void*>fid)
    elif dims == 2:
        return create_2d_py_func_wrapper(<void*>fid)
    else:
        return create_3d_py_func_wrapper(<void*>fid)

# -------------------------------------------------------------
# Python-accessible built-in test functions
# -------------------------------------------------------------
def py_recursive_fractal_cliff_valley(*args):
    if len(args) == 1:
        return recursive_fractal_cliff_valley(args[0], 0.0, 0.0)
    elif len(args) == 2:
        return recursive_fractal_cliff_valley(args[0], args[1], 0.0)
    return recursive_fractal_cliff_valley(args[0], args[1], args[2])

def py_rosenbrock_3d(*args):
    if len(args) == 1:
        return rosenbrock_3d(args[0], 0.0, 0.0)
    elif len(args) == 2:
        return rosenbrock_3d(args[0], args[1], 0.0)
    return rosenbrock_3d(args[0], args[1], args[2])

def py_sphere_function(*args):
    if len(args) == 1:
        return sphere_function(args[0], 0.0, 0.0)
    elif len(args) == 2:
        return sphere_function(args[0], args[1], 0.0)
    return sphere_function(args[0], args[1], args[2])

def py_ackley_function(*args):
    if len(args) == 1:
        return ackley_function(args[0], 0.0, 0.0)
    elif len(args) == 2:
        return ackley_function(args[0], args[1], 0.0)
    return ackley_function(args[0], args[1], args[2])

def py_rastrigin_function(*args):
    if len(args) == 1:
        return rastrigin_function(args[0], 0.0, 0.0)
    elif len(args) == 2:
        return rastrigin_function(args[0], args[1], 0.0)
    return rastrigin_function(args[0], args[1], args[2])

# -------------------------------------------------------------
# Main optimization interface
# -------------------------------------------------------------
def daps_cython(func, bounds=None, options=None):
    default_opts = {'dimensions':3, 'min_prime_idx':5, 'max_iterations':1000, 'tolerance':1e-8, 'callback':None, 'verbose':False}
    opts = default_opts.copy()
    if options: opts.update(options)

    cdef int dims = opts['dimensions']
    if dims < 1 or dims > 3:
        raise ValueError("Dimensions must be 1,2 or 3")
    if bounds is None or len(bounds) != 2*dims:
        raise ValueError(f"Bounds must be length {2*dims}")

    cdef double x_min = bounds[0], x_max = bounds[1]
    cdef double y_min = 0.0, y_max = 0.0, z_min = 0.0, z_max = 0.0
    if dims >= 2: y_min, y_max = bounds[2], bounds[3]
    if dims == 3: z_min, z_max = bounds[4], bounds[5]

    cdef int max_iter = opts['max_iterations'], min_idx = opts['min_prime_idx']
    cdef double tol = opts['tolerance']
    cdef void* cb_ptr = NULL
    if opts['callback'] is not None:
        cid = id(opts['callback'])
        _callback_registry[cid] = opts['callback']
        cb_ptr = <void*>cid

    cdef function[double(double,double,double)] cpp_func
    if isinstance(func, str):
        name = func.lower()
        if name == 'recursive_fractal_cliff_valley': cpp_func = recursive_fractal_cliff_valley
        elif name == 'rosenbrock_3d': cpp_func = rosenbrock_3d
        elif name == 'sphere_function': cpp_func = sphere_function
        elif name == 'ackley_function': cpp_func = ackley_function
        elif name == 'rastrigin_function': cpp_func = rastrigin_function
        else: raise ValueError(f"Unknown builtin: {func}")
    else:
        py = func.func if hasattr(func, 'func') else func
        if not callable(py): raise ValueError("func must be callable or builtin name")
        cpp_func = create_function_wrapper(py, dims)

    cdef DAPSResult res
    with nogil:
        res = daps_optimize(cpp_func, x_min, x_max, y_min, y_max, z_min, z_max,
                             max_iter, min_idx, min_idx, min_idx, cb_ptr, tol, dims)

    if cb_ptr: _callback_registry.pop(<long>cb_ptr, None)

    result = {
        'x': [res.x[i] for i in range(dims)], 'fun': res.fun_val,
        'nfev': res.nfev, 'nit': res.nit, 'success': bool(res.success),
        'dimensions': dims, 'final_prime_idx_x': res.final_prime_idx_x
    }
    if dims >= 2: result['final_prime_idx_y'] = res.final_prime_idx_y
    if dims == 3: result['final_prime_idx_z'] = res.final_prime_idx_z
    return result

