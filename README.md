# Marching Squares parallelo su GPU

## Implementazione Marching Squares usata attualmente

### Libreria Skimage implementazione find_contours
_find_contours.py (https://github.com/scikit-image/scikit-image/blob/main/skimage/measure/_find_contours.py)	
_find_contours_cy.pyx (https://github.com/scikit-image/scikit-image/blob/main/skimage/measure/_find_contours_cy.pyx)
	

## Potenziali strumenti per la parallelizzazione

### nvc++
Compilazione codice Cython tramite nvc++ con flag per la parallelizzazione su GPU.
(Nvidia supporto a Cython e parallelizzazione automatica su CPU e GPU https://developer.nvidia.com/blog/accelerating-python-on-gpus-with-nvc-and-cython/)

### API Cuda-Python

#### API Cuda-Python Nvidia (>=11.4)
https://nvidia.github.io/cuda-python/overview.html

#### Numba
https://numba.readthedocs.io/en/stable/cuda/index.html


## Current state
|               |    PC    |   server   |
| ------------- | -------- | ---------- |
| jupyter_test (aborted) |    ×     |     ×      |
| nvc++ (aborted)		|    ×     |     ×      |
| api_cuda_test |    ✓    | 	 ✓	|
| api_cuda	|    ✓    |     ✓      |

