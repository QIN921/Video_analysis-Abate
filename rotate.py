import numpy as np
import math


def rotate(a, b, alpha, k):
    pi = math.pi
    cos = math.cos(alpha/180*pi)
    sin = math.sin(alpha/180*pi)
    A = cos*cos/a/a+sin*sin/b/b
    C = sin*sin/a/a+cos*cos/b/b
    C = C/k/k
    B = 2*cos*sin*(1/a/a-1/b/b)
    B = B/k
    deta = math.sqrt((A-C)*(A-C)+B*B)
    a = math.sqrt(2/(A+C-deta))
    b = math.sqrt(2/(A+C+deta))
    theta = math.atan((C-A-deta)/B)*180/pi
    print(a)
    print(b)
    print(theta)


if __name__ == '__main__':
    rotate(0.2, 0.11, np.random.rand(1)*360, 0.2)
