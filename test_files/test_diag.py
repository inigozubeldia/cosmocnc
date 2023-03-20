import numpy as np
import time

t00 = time.time()

M = np.array([[1,2],[4,-1]])

w,A = np.linalg.eig(M)

print(w)
print(A)


A_inv = np.linalg.inv(A)


t0 = time.time()

print(t0-t00)


M_diag = np.matmul(A_inv,np.matmul(M,A))

print("Time",time.time()-t0)
t1 = time.time()

M_diag_2 = np.einsum("ij,jk->ik",A_inv,np.einsum("ij,jk->ik",M,A))

print(M_diag_2)
print(M_diag)
print(time.time()-t1)
