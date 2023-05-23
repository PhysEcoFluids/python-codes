import numpy as np
import matplotlib.pyplot as plt

N = 128

x1 = [0]
y1 = [1.e3]

x2 = [3e3]
y2 = [1e3]

x3 = np.linspace(3e3, 5e3, N)
y3 = 1e3 - 250/np.cosh(((x3-4e3)/200)**2)

x4 = [8e3]
y4 = [1e3]

y5 = [0]
x5 = [8.e3]

x6 = [0]
y6 = [0]


x = np.r_[x1, x2, x3, x4, x5, x6]
y = np.r_[y1, y2, y3, y4, y5, y6]

# Produce gmsh geo file:


print(f"Point({1}) = {{{x[0]}, {y[0]}, 0.0, {150}}};")
print(f"Point({2}) = {{{x[1]}, {y[1]}, 0.0, {50}}};")

clen = 50.0
for i in range(2, N+2):
    print(f"Point({i+1}) = {{{x[i]}, {y[i]}, 0.0, {clen}}};")
    
print(f"Point({N+3}) = {{{x[N+2]}, {y[N+2]}, 0.0, {150}}};")
print(f"Point({N+4}) = {{{x[N+3]}, {y[N+3]}, 0.0, {150}}};")
print(f"Point({N+5}) = {{{x[N+4]}, {y[N+4]}, 0.0, {150}}};")


print(f"Line(1) = {{1, 2}};")

seq1 = ", ".join([str(i) for i in range(2, N+3)])
print(f"Spline(2) = {{{seq1}}};")

#seq2 = ", ".join([str(i) for i in range(127, 127+16+1)])
print(f"Line(3) = {{{N+2}, {N+3}}};")
print(f"Line(4) = {{{N+3}, {N+4}}};")
print(f"Line(5) = {{{N+4}, {N+5}}};")
print(f"Line(6) = {{{N+5}, 1}};")

print("Curve Loop(1) = {1, 2, 3, 4, 5, 6};")
print("Plane Surface(1) = {1};")

#seq3 = ", ".join([str(i) for i in range(127+16, 127+32+1)])
#print(f"Spline(3) = {{{seq3}}};")

#seq4 = ", ".join([str(i) for i in range(129+31, 129+48)]) + ", 1"
#print(f"Spline(4) = {{{seq4}}};")

#plt.figure()
#plt.plot(x, y, '.-')
#plt.draw()
#plt.show()
