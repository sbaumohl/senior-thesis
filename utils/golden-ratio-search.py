import math

import numpy as np

GOLDEN_RATIO = (1 + math.sqrt(5)) * 0.5

a, b = 0, 2 * math.pi
f = lambda x: 2 * x * math.sin(x)

while abs(a - b) > 0.00001:
    print(a, b)
    k = GOLDEN_RATIO * (b - a)
    x1, x2 = a + k, b - k

    if f(x1) < f(x2):
        a = x1
    else:
        b = x2
