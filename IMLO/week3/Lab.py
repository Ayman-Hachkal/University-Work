import numpy as np

def Q1(): 
    x = np.matrix([[2], [3], [5]])
    y = np.matrix([[3], [0], [4]])
    a = x.T * y 
    b = x * y.T

    print("A: ", a)
    print("B: ", b)
    
def Q2():
    A = np.matrix([[4, 1, 0, 2],
                   [2, 0, 1, 3]])
    b = np.matrix([[3],
                   [0],
                   [5],
                   [1]])
    c = A*b
    print(c)

def Q3():
    A = np.matrix([[4, 1, 0, 2],
                   [2, 0, 1, 3]])
    B = np.matrix([[3, 2, 0],
                   [0, 4, 2],
                   [5, 0, 1],
                   [1, 3, 4]])
    C = A * B
    print("Dimensions of A: ", A.shape)
    print("Dimensions of B: ", B.shape)
    print(C)

def Q4():
    w = np.matrix([[2],
                   [4],
                   [3],
                   [1]])

    wnorm = np.linalg.norm(w)
    print(wnorm)

def Q5():
    u = np.matrix([[5],
                   [0],
                   [0]])
    unitnorm = u/np.linalg.norm(u)
    print(unitnorm)

def B(): 
    B = np.matrix([[3, 2, 0],
                   [0, 4, 2],
                   [5, 0, 1],
                   [1, 3, 4]])
    print(B[1:3, 1])

def dot(v1, v2):
    vdot = sum(v1 * v2)
    return vdot

def C():
    x = np.array([2, 3, 5])
    v = np.array([3, 0, 4])
    print(dot(x, v))

def transform(data):
    return (np.dot(np.array([1, 1, 1]), data.T)) #1x3 * 3xi

def DandE():
    data = np.matrix([[1, 17, 9],
                      [2, 21, 10],
                      [3, 12, 17]])
    print(transform(data))

def F():
    data = np.matrix([[17, 9, 1],
                      [21, 10, 1],
                      [12, 17, 1]])
    theta = np.array([4, 5, -100])
    
    print(np.dot(data, theta.T))



#print("QA")
#Q1()
#
#print("QB")
#Q2()
#
#print("QC")
#Q3()
#
#print("QD")
#Q4()
#
#print("QE")
#Q5()

B()
C()
DandE()
F()

