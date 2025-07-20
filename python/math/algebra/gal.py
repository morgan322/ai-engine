import numpy as np
from sympy import symbols, expand, Poly, QQ, sqrt, Integer
from sympy.polys.galoistools import gf_irreducible_p
from sympy.polys.domains import ZZ
from sympy.ntheory import factorint
from sympy.combinatorics import Permutation, PermutationGroup

class Polynomial:
    def __init__(self, coefficients):
        """初始化多项式，系数按降序排列"""
        self.coefficients = coefficients
        self.degree = len(coefficients) - 1
        self.x = symbols('x')
        self.poly = sum(coef * self.x**(self.degree - i) for i, coef in enumerate(coefficients))
        self.sympy_poly = Poly(self.poly, self.x, domain=QQ)
        
    def __str__(self):
        return str(self.poly.expand())
    
    def discriminant(self):
        """计算多项式的判别式"""
        return self.sympy_poly.discriminant()
    
    def is_irreducible(self):
        """判断多项式在有理数域上是否不可约"""
        # 对于2-4次多项式，检查是否有有理根
        if self.degree == 2 or self.degree == 3:
            # 有理根定理：可能的有理根为p/q，其中p是常数项的因子，q是首项系数的因子
            constant_term = self.coefficients[-1]
            leading_coef = self.coefficients[0]
            
            if constant_term == 0:
                return False  # 有根x=0
                
            factors_p = list(factorint(abs(constant_term)).keys())
            factors_q = list(factorint(abs(leading_coef)).keys())
            
            possible_roots = [p/q for p in factors_p for q in factors_q] + \
                            [-p/q for p in factors_p for q in factors_q] + \
                            factors_p + [-p for p in factors_p]
            
            for root in possible_roots:
                if self.evaluate(root) == 0:
                    return False
            return True
        elif self.degree == 4:
            # 对于4次多项式，使用sympy的判断
            # 修复：确保所有系数都是SymPy整数类型
            coeffs = [Integer(c) for c in self.sympy_poly.all_coeffs()]
            domain = ZZ
            # 转换为首一多项式（最高次系数为1），以便在整数环上检查不可约性
            if coeffs[0] != 1:
                leading = coeffs[0]
                coeffs = [c // leading for c in coeffs]
            return gf_irreducible_p(coeffs, domain, 1)  # 1表示多项式的变量数量
        return True
    
    def evaluate(self, x_val):
        """计算多项式在x_val处的值"""
        return self.sympy_poly.subs(self.x, x_val)
    
    def galois_group(self):
        """确定多项式的伽罗瓦群"""
        if self.degree == 1:
            return "C1 (平凡群)"
        
        if self.degree == 2:
            # 二次多项式ax²+bx+c的伽罗瓦群
            if self.is_irreducible():
                return "C2 (二阶循环群)"
            else:
                return "C1 (平凡群)"
                
        elif self.degree == 3:
            # 三次多项式的伽罗瓦群分析
            if not self.is_irreducible():
                return "C1 (平凡群) 或 C2 (如果分解为一次和二次不可约因子)"
            
            D = self.discriminant()
            # 检查判别式是否为完全平方
            sqrt_D = sqrt(D)  # 使用sympy.sqrt()函数
            if sqrt_D.is_integer or (sqrt_D.is_rational and sqrt_D.as_integer_ratio()[1] == 1):
                return "A3 (三阶循环群)"
            else:
                return "S3 (对称群)"
                
        elif self.degree == 4:
            # 四次多项式的伽罗瓦群分析
            if not self.is_irreducible():
                factors = self.sympy_poly.factor_list()[1]
                degrees = [f[0].degree() for f in factors]
                
                if degrees == [1, 3]:  # 一次和三次
                    return self.galois_group_cubic(factors[1][0].coeffs())
                elif degrees == [2, 2]:  # 两个二次
                    # 检查两个二次因子的伽罗瓦群
                    gal1 = Polynomial(factors[0][0].coeffs()).galois_group()
                    gal2 = Polynomial(factors[1][0].coeffs()).galois_group()
                    
                    if gal1 == "C1" and gal2 == "C1":
                        return "C1 × C1 (平凡群)"
                    elif (gal1 == "C1" and gal2 == "C2") or (gal1 == "C2" and gal2 == "C1"):
                        return "C2 (二阶循环群)"
                    else:  # 两个C2
                        return "V (克莱因四元群)"
                elif degrees == [1, 1, 2]:  # 两个一次和一个二次
                    return Polynomial(factors[2][0].coeffs()).galois_group()
                else:  # 四个一次
                    return "C1 (平凡群)"
            
            # 不可约四次多项式的伽罗瓦群分析
            # 这里简化处理，只返回可能的群
            return "S4, A4, D4, C4, 或 V (需进一步分析预解三次多项式)"
        
        return "未知"
    
    def galois_group_cubic(self, coeffs):
        """辅助函数：计算三次多项式的伽罗瓦群"""
        cubic_poly = Polynomial(coeffs)
        return cubic_poly.galois_group()
    
    def is_solvable_by_radicals(self):
        """判断多项式是否可用根式求解"""
        group = self.galois_group()
        
        # 所有2-4次多项式都是可解的
        if self.degree <= 4:
            return True
        
        # 对于更高次多项式，需要检查伽罗瓦群是否为可解群
        # 这里仅作为示例，实际需要更复杂的群论分析
        solvable_groups = ["C1", "C2", "C3", "A3", "C4", "D4", "V", "S4", "A4"]
        for solvable_group in solvable_groups:
            if solvable_group in group:
                return True
        return False

# 测试示例
if __name__ == "__main__":
    # 二次多项式 x² - 2
    poly2 = Polynomial([1, 0, -2])
    print(f"多项式: {poly2}")
    print(f"判别式: {poly2.discriminant()}")
    print(f"伽罗瓦群: {poly2.galois_group()}")
    print(f"是否可用根式求解: {poly2.is_solvable_by_radicals()}")
    print()
    
    # 三次多项式 x³ - 3x + 1
    poly3 = Polynomial([1, 0, -3, 1])
    print(f"多项式: {poly3}")
    print(f"判别式: {poly3.discriminant()}")
    print(f"是否不可约: {poly3.is_irreducible()}")
    print(f"伽罗瓦群: {poly3.galois_group()}")
    print(f"是否可用根式求解: {poly3.is_solvable_by_radicals()}")
    print()
    
    # 四次多项式 x⁴ - 5x² + 6
    poly4 = Polynomial([1, 0, -5, 0, 6])
    print(f"多项式: {poly4}")
    print(f"判别式: {poly4.discriminant()}")
    print(f"是否不可约: {poly4.is_irreducible()}")
    print(f"伽罗瓦群: {poly4.galois_group()}")
    print(f"是否可用根式求解: {poly4.is_solvable_by_radicals()}")    