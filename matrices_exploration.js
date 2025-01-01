// https://mljs.github.io/matrix/
// Note, all functions & variables (such as #initData) in matrix.js which previously had a # had the symbol replaced by __ to
// fix a crash when using browserify.

const {
    Matrix,
    inverse,
    solve,
    linearDependencies,
    // QrDecomposition,
    // LuDecomposition,
    // CholeskyDecomposition,
    EigenvalueDecomposition,
    pseudoInverse,
  } = require('ml-matrix');
  
function print(val){
    console.log(String(val));
}
const ones_matrix = Matrix.ones(3, 3);
print('ones_matrix: ' + ones_matrix)
var A = new Matrix([
    [ 1,  1],
    [-1, -1],
  ]);

var exp_A = Matrix.exp(A);

// abs, acos, acosh, asin, asinh, atan, atanh, cbrt, ceil, clz32, cos, cosh, exp, expm1, floor, fround, log, log1p, log10, log2, round,
// sign, sin, sinh, sqrt, tan, tanh, trunc

print("exp_A: " + exp_A);
let expA_minus_one = exp_A.sub(1);
print("expA_minus_one: " + expA_minus_one);
let expA_times_A = exp_A.mul(A);
print("expA_times_A: " + expA_minus_one);

var A2 = new Matrix([
  [3,    1],
  [4.25, 1],
  [5.5,  1],
  [8,    1],
]);

//least squares solution, find x2 in the matrix problem A2 * x2 = B2
var B2 = Matrix.columnVector([4.5, 4.25, 5.5, 5.5]);
var x2 = solve(A2, B2);
var error = Matrix.sub(B2, A2.mmul(x2)); // The error enables to evaluate the solution x found.

x2_transp = x2.transpose();

print(x2_transp);

var C = new Matrix([
  [2, 3, 5],
  [4, 1, 6],
  [1, 3, 0],
]);

pinvC = pseudoInverse(C);
invC = inverse(C);

let unity1 = C.mmul(invC);
let unity2 = C.mmul(pinvC);

print('matrix times its inverse: ' + unity1)
print('matrix times its pseudoinverse: ' + unity2)

max1 = Matrix.max(C, invC);   //elementwise maximum

print('elementwise maximum: '+ max1)


var e = new EigenvalueDecomposition(C);
var real = e.realEigenvalues;
var imaginary = e.imaginaryEigenvalues;
var vectors = e.eigenvectorMatrix;

print('real eigenvalues: ' + real + '\nimaginary eigenvalues: ' + imaginary);
print('eigenvectors: ' + vectors);


var D = new Matrix([
  [2, 0, 0, 1],
  [0, 1, 6, 0],
  [0, 3, 0, 1],
  [0, 0, 1, 0],
  [0, 1, 2, 0],
]);

var dependencies = linearDependencies(D);
// dependencies is a matrix with the dependencies of the rows. 
// When we look row by row, we see that the first row is [0, 0, 0, 0, 0], // so it means that the first row is independent, 
// and the second row is [ 0, 0, 0, 4, 1 ], i.e the second row = 4 times the 4th row + the 5th row.

print('linear dependency matrix: ' + dependencies);

norm1 = C.norm();

print('Frobenius (L2) norm: ' + norm1);
x = 2;
