

def gcd_recursive(a, b):
    if b == 0:
        return a
    else:
        return gcd_recursive(b, a % b)
    
def gcd_iterative(a, b):
    while b != 0:
        a, b = b, a % b
    return a

a = 3143
b = 16146515
print("a,b:",a,b)
result = gcd_recursive(a, b)
print("gcd_recursive: a 和 b 的最大公约数是:", result)

result = gcd_iterative(a, b)
print("gcd_iterative: a 和 b 的最大公约数是:", result)

