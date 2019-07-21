using MendelIHT
using Random
using GLM
using Distributions
using DataFrames

Random.seed!(2019)

#define problem size and distribution
n = 1000
p = 10000
k = 10
d = Bernoulli
l = canonicallink(d())

#simulate covariates and true model
X = randn(n, p)
Z = ones(n, 1)
b = zeros(p)
b[1:k] .= randn(k)
shuffle!(b)
correct_position = findall(!iszero, b)

#simulate response
prob = linkinv.(l, X * b)
y = [rand(d(i)) for i in prob]
y = Float64.(y)

#run IHT
result = L0_reg(X, Z, y, 1, k, Bernoulli(), LogitLink())

#check result
compare_model = DataFrame(
    true_β      = b[correct_position], 
    estimated_β = result.beta[correct_position])
