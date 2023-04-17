# distutils: language=c++

def get_hello():
    k=0
    for i in range(10):
        for j in range(10):
            k=i*j

    return k
