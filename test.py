import numpy as np

s = np.array([1,3,5,6,6,7,7,7])
g = np.array([0,0,0,1,2,5,6,7])
match = np.zeros(8)
r = 7
z = 7
while r >= 0:
    if s[r] < g[z]:
        z -= 1
    elif s[r] == g[z]:
        if g[z] == g[z - 1]:
            z -= 1
        else:
            match[r] = z
            r -= 1
    else:
        match[r] = z
        r -= 1

print(match)