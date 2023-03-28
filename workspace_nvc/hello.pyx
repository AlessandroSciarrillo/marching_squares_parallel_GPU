# distutils: language=c++

def get_hello():
    k=0
    for i in range(1000):
        for j in range(10000):
            k=i*j

    return k
