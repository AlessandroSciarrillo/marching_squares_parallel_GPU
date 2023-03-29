
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
    
* Parallelizzando solo i due for di find_contours_cy il progetto non è troppo piccolo?
    * Cosa aggiungere?
    * Includere nel progetto il codice dello stage Aliquis?
        * Parallelizzare anche il ciclo sui vari canali

Riunione 01/03/2023
- utilizzare compilatore nvc++ ottimizzato da nvidia per compilare find_contours_cy in Cython
- fare benchmark rispetto a versione utilizzata ora da loro su Aliquis (versione vanilla)
- nel caso provare a scrivere kernel cuda con Python-Cuda riscrivendo find_contours_cy


# Current
Errore segmentation fault (core dump)
- generato all'import del codice compilato con nvc++ (CC=nvc++ python setup.py build_ext --inplace)
	- se si compila con cython invece il codice esegue correttamente (python setup.py build_ext --inplace)

## Next
- tentare di capire la causa del segmentation fault generato dal import su python del cython compilato con nvc++
- capire perchè le prestazioni sul loro Jupyter notebook con la versione GPU siano uguali a quelle con CPU seriale
- fare test con CUDA API

- scaricare docker nvidia con tutto l ambiente
- proseguire con nvc++, controllare i flag, testare codice su RTX2060
