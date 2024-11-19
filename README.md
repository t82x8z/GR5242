java c
GR5242 HW03 Problem 4: Automatic Differentiation
Instructions: This problem is an individual assignment -- you are to complete this problem on your own, without conferring with your classmates. You should submit a completed and published notebook to Courseworks; no other files will be accepted.
Description:
This homework adds more detail to the autodiff lecture in class. There are seven questions (30 points) in total, which include coding and written questions. You can only modify the codes and text within ### YOUR CODE HERE ### and/or ### YOUR ANSWER HERE ###.In [ ]:import torchimport matplotlib.pyplot as pltimport numpy as npfrom datetime import datetimeimport sympy as sym
Differentiation
Different ways of differentiation: In this problem we consider a particular function (which we call it an op in the code block, short for operation)  f : R → R : x ↦ 4x(1 − x) and it's compositions with itself. Let us define,

where f (n)(x) = f(f(n−1)(x)) is the n times composited function f with itself. The goal of this problem is to explore various ways of differentiation, i.e. dx/d gn(x) which are listed below:
Numerical differentiation
Symbolic differentiation
Autodiff in forward or reverse modeIn [ ]:def op(inp):return 4 * inp * ( 1 - inp )def operation(inp, n = 1):temp = inpfor i in range(n):temp = op(temp) # apply this operation n timesreturn temp
Numerical Differentiation
Based on the definition of derivative at a particular point x0,

The formula above suggests a simple way of approximating the derivative by taking Dhgn(x0) for a particular choice of h as your approximate derivative. This is also known as the finite difference method. Note that this approach only requires evaluation of your function around the point you are trying to take the derivative at so it's computationally efficient but the caveat is that choosing a proper h to obtain good enough approximations is generally hard (specially when the function is multivariate). However, in our case we have some structure over the function we are trying to take derivative of.
Question 1 (3 points): Use a finite difference with tolerance 1e-12 to approximate the derivative of at g3(x) at x = 1:In [ ]:tol = 1e-12### Your Code Here ###
The approximation will be bad at certain points x, and becomes less stable as n becomes larger for a fixed h (or as the dimensionality of the function grows).
Calculus and (manual/symbolic) differentiation
Notice that f is a polynomial and that composition of polynomials yields also a polynomial. Therefore, gn is a polynomial with degree 2n  (try to argue this for yourself using induction); we can compute the derivative using calculus.
Instead of computing the derivative by hand we use the help of an auxillary package sympy and try to compute the derivative. This package uses symbolic variables and traces operations such as add, multiplication, division, etc., applied onto these variables and computes the derivative using chain rule.
It is not difficult to (manually) derive a closed form. expression for gn using the recursive formula
gn (x) = 4gn−1(x)(1 − gn−1(x)),  g1 (x) = 4x(1 − x).
The following block of code prints this closed form. expression in terms of a symbolic variable X.In [ ]: X = sym.Symbol('X') # Create a symbolic variablesym.init_printing(use_unicode=False, wrap_line=True)
Now that we have the closed form. expression, we may use the chain rule to express the derivative of gn.In [ ]:dydx_symbolic = sym.diff(Y, X)dydx_symbolicOut[ ]:64X (1 − X)(8X − 4)(−16X (1 − X)(−4X (1 − X) + 1) + 1)+ 64X (1 − X)(−4X (1 − X) + 1)(−16X (1 − X)(8X − 4) + 16X (−4X (1 − X) + 1)⋅ (1 − X)(−4X (1 − X) + 1)) − 64X (−4X (1 − X) + 1)(−16X (1 − X)(−4X (1 − X)⋅ (1 − X)(−4X (1 − X) + 1)(−16X (1 − X)(−4X (1 − X) + 1) + 1)
Question 2 (3 po代 写GR5242 HW03 Problem 4: Automatic DifferentiationR
代做程序编程语言ints): Compute the exact derivative of at by evaluating the symbolic expression.In [ ]:# hint: read about subs/evalf here: https://docs.sympy.org/latest/modules/numer### Your Code Here ###
As it is evident the closed form. expression for dydx_symbolic is unwieldy. We can make it more efficient by expanding and collecting all the terms. The goal is to represent the derivative as a natural polynomial. Yet this is another way of computing the derivative.
Question 3 (3 points): Using the sympy documentation page expand/simplify the closed form. expression of gn(x) and print its derivative as a symbolic expression.In [ ]:### Your Code Here ###polynomial = ...dydx_symbolic_simplified = ...dydx_symbolic_simplified # This should express a polynomial of degree 2^n - 1.
EllipsisOut[ ]:
Now that we have a compact closed form. for the derivative we can evaluate it efficiently at various values of x.
Question 4 (2 points): Evaluate dydx_symbolic_simplified at x = 1.In [ ]:### Your Code Here ###
Chain Rule and Autodiff
As we've seen in the previous part, symbolic engines computes the derivative using some closed form. expression of the function. In particular, sympy simplified the recursive operation and expressed the function gn in terms of basic operations over the symbolic variable X. On the other hand, Autodiff does not necessarily simplify these operations and apply chain rule directly to the latest operation either in forward or backward mode.
Question 5 (4 points): Using the recursive representation gn (x) = f(gn−1(x)) calculate the derivative of gn as a function of gn−1 and g'n−1. (To be clear, this part is not code; write the mathematical expression below)
'### Your Answer Here ###'
This implies that we can compute g'n(x) in the forward mode if we could compute gn−1(x) and g'n−1 (x) in the forward mode. Indeed, this is possible by applying the same logic to g'n−1 (x) and compute it based on gn−2(x) and g'n−2 (x); In other words, by augmenting the forward computation graph g1 → g2 ⋯ → gn with (g1,g'1) → (g2,g'2), ⋯ → (gn,g'n) we can compute the derivative in a forward pass.
Question 6 (8 points): Modify the functions  op  and  operation with their counterparts op_with_grad  and  operation_with_grad  using the logic explained above to compute the derivative of gn. Check your function by evaluating it atx = 1.In [ ]:### Your Code Here ###def op_with_grad(inp, grad):# inp could be any function of x.# grad is the corresponsing derivative of inp with respect to x.# The function should return a tuple where the first element is the `op` appl# and the second element should be the gradient of op(inp) with respect to x.passdef operation_with_grad(inp, n = 1):# inp represents x here.# This function should output a tuple where the first element is the value of# and the second element should be the derivative evaluated at inp, i.e. g_n^passoperation_with_grad(1.0, n = 3)
Fortunately, torch can do last part for us! As was discussed in class, torch interprets our code and builds up a computation graph using operations that knows their gradients already and complements each operation with their backward gradient; in order to compute the gradient it follows the following backwards path
where g0(x) = x is the identity function.
Question 7 (7 points): Use PyTorch to calculate the derivative of g_n(x) with respect to xIn [ ]:def torch_op(inp):return 4 * inp * ( 1 - inp )def torch_operation(inp, n = 1):temp = inpfor i in range(n):temp = torch_op(temp) # apply this operation n timesreturn tempIn [ ]:x = torch.tensor(1.0, requires_grad=True)#### Your code here ###### define the function with n=3 iterations as before, and use pytorch to autodifdf_dx =...print(df_dx.numpy())In [ ]:











         
加QQ：99515681  WX：codinghelp  Email: 99515681@qq.com
