
# Libreria Skimage implementazione find_contours
https://github.com/scikit-image/scikit-image/blob/main/skimage/measure/_find_contours.py
https://github.com/scikit-image/scikit-image/blob/main/skimage/measure/_find_contours_cy.pyx

## Nvidia supporto a Cython e parallelizzazione automatica su CPU e GPU 
https://developer.nvidia.com/blog/accelerating-python-on-gpus-with-nvc-and-cython/



# Nvidia (>=11.4) Cuda-Python
https://nvidia.github.io/cuda-python/overview.html
	
                                C++         Python
    Application execution       1076ms      1080ms    (-0.004%)

## Numba (consigliato da Emanuele)
https://numba.readthedocs.io/en/stable/cuda/index.html


# Next steps
Utilizzare find_countours.py sostituendo l'esecuzione del metodo 
_get_contour_segments importato da find_contours_cy.pyx (Cython)
con una versione di _get_contour_segments riscritta in C++ che 
utiizzerà le CUDA API o nativamente o tramite Numba.

Alternativa:
scrivere una propria versione di find_contours 
(senza partire dalla versione skimage) che prenda gli stessi
argomenti in input ma che implementi una versione più efficente di
Marching Squares(*se trovata) interamente scritta per
funzionare con CUDA API o Numba.

Valutazioni:
* Trovare versione ottimizzata Marching Cubes con CUDA? hanno senso?
    https://ieeexplore.ieee.org/document/9415927

* Utilizzare Marching Cubes/Blocks? 
    Speedup Marching Blocks PMB è applicabile al 2D?







