# -*- coding: utf-8 -*-
"Bernoulli numbers"
from fractions import Fraction

B=[Fraction(1),Fraction(-1,2)]

def factorial(n):
    if n==0:
        return 1
    return n*factorial(n-1)

def binom(n,k):
    return Fraction(factorial(n),factorial(k)*factorial(n-k))

def next_bernoulli():
    r=len(B)    # Assume r>1
    s=0
    if r%2:
        return s
    for m in range(r):
        s+=binom(r,m)*(-1)**m*B[m]/(r+1-m)
    return 1-s

N=10

for i in range(N-2):
    B.append(next_bernoulli())

def to_latex(i,fr):
    sign="-"
    if fr>=0:
        sign=""
    denom="}"
    if fr.denominator!=1:
        denom=r"\over%d}" % fr.denominator
    return r"B_{%d}&=%s{%d%s \\" % (i,sign,abs(fr.numerator),denom)

print(r"\begin{aligned}")
for i in range(N):
    print(to_latex(i,B[i]))
print(r"\end{aligned}")