#https://vimsky.com/examples/detail/python-method-scipy.optimize.linprog.html   参考资料   总结了20种用法
from  scipy import optimize as opt
import  numpy as np
import pandas as pd

# Import lib
def solve_linprog(self, nu):
    n_hs = len(self.hs)
    n_constraints = len(self.constraints.index)
    if self.last_linprog_n_hs == n_hs:
        return self.last_linprog_result
    c = np.concatenate((self.errors, [self.B]))
    A_ub = np.concatenate((self.gammas - self.eps, -np.ones((n_constraints, 1))), axis=1)
    b_ub = np.zeros(n_constraints)
    A_eq = np.concatenate((np.ones((1, n_hs)), np.zeros((1, 1))), axis=1)
    b_eq = np.ones(1)
    result = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='simplex')
    Q = pd.Series(result.x[:-1], self.hs.index)
    dual_c = np.concatenate((b_ub, -b_eq))
    dual_A_ub = np.concatenate((-A_ub.transpose(), A_eq.transpose()), axis=1)
    dual_b_ub = c
    dual_bounds = [(None, None) if i == n_constraints else (0, None) for i in range(n_constraints + 1)]  # noqa: E501
    result_dual = opt.linprog(dual_c,
                              A_ub=dual_A_ub,
                              b_ub=dual_b_ub,
                              bounds=dual_bounds,
                              method='simplex')
    lambda_vec = pd.Series(result_dual.x[:-1], self.constraints.index)
    self.last_linprog_n_hs = n_hs
    self.last_linprog_result = (Q, lambda_vec, self.eval_gap(Q, lambda_vec, nu))
    return self.last_linprog_result

print()