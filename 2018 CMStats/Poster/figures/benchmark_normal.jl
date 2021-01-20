#load packages
using IHT
using SnpArrays
using DataFrames
using Distributions
using StatsFuns: logistic
using BenchmarkTools

function normal_response(
    n :: Int64,   # number of cases
    p :: Int64,   # number of predictors
    k :: Int64,   # number of true predictors per group
    s :: Float64  # noise vector, from very little noise to a lot of noise
)
    #construct snpmatrix, covariate files, and true model b
    x           = SnpArray(rand(0:2, n, p))    # a random snpmatrix
    z           = ones(n, 1)                   # non-genetic covariates, just the intercept
    true_b      = zeros(p)                     # model vector
    true_b[1:k] = randn(k)                     # Initialize k non-zero entries in the true model
    shuffle!(true_b)                           # Shuffle the entries
    correct_position = find(true_b)            # keep track of what the true entries are
    noise = rand(Normal(0, s), n)              # noise

    #compute mean and std used to standardize data to mean 0 variance 1
    mean_vec, minor_allele, = summarize(x)
    update_mean!(mean_vec, minor_allele, p)
    std_vec = std_reciprocal(x, mean_vec)

    #simulate phenotypes under different noises by: y = Xb + noise
    y_temp = zeros(n)
    SnpArrays.A_mul_B!(y_temp, x, true_b, mean_vec, std_vec)
    y = y_temp + noise

    #define IHTVariable needed to run L0_reg
    v = IHTVariables(x, z, y, 1, k)
    
    #benchmark the result and return
    b = @benchmarkable L0_reg($v, $x, $z, $y, 1, $k) seconds=500 samples=10
    return run(b)
end

#set random seed
srand(2018) 

#how many different sample sizes to run?
num_samples = 10

#problem dimension?
samples = [1000i for i in 1:num_samples]
p = 100000
k = 10
s = 0.1

#benchmark result on each sample size and store benchmark result
normal_result = Vector{BenchmarkTools.Trial}(num_samples)
for i in 1:length(samples)
     normal_result[i] = normal_response(samples[i], p, k, s)
     println("completed " * string(i))
end

#set random seed
srand(2018) 

#how many different sample sizes to run?
num_samples = 10

#problem dimension?
samples = [1000i for i in 1:num_samples]
p = 100000
k = 10
s = 0.1

#benchmark result on each sample size and store benchmark result
logistic_result = Vector{BenchmarkTools.Trial}(num_samples)
for i in 1:length(samples)
     logistic_result[i] = logistic_response(samples[i], p, k)
     println("completed " * string(i))
end

function poisson_response(
    n :: Int64,   # number of cases
    p :: Int64,   # number of predictors
    k :: Int64,   # number of true predictors per group
)
    #construct snpmatrix, covariate files, and true model b
    x           = SnpArray(rand(0:2, n, p))    # a random snpmatrix
    z           = ones(n, 1)                   # non-genetic covariates, just the intercept
    true_b      = zeros(p)                     # model vector
    true_b[1:k] = randn(k)                     # Initialize k non-zero entries in the true model
    shuffle!(true_b)                           # Shuffle the entries
    correct_position = find(true_b)            # keep track of what the true entries are

    #compute mean and std used to standardize data to mean 0 variance 1
    mean_vec, minor_allele, = summarize(x)
    update_mean!(mean_vec, minor_allele, p)
    std_vec = std_reciprocal(x, mean_vec)

    #simulate phenotypes
    y_temp = zeros(n)
    SnpArrays.A_mul_B!(y_temp, x, true_b, mean_vec, std_vec)
    
    # Apply inverse log link and sample from poisson distribution with given mean
    y = zeros(n)
    y_temp = exp.(y_temp)
    for i in 1:n
        dist = Poisson(y_temp[i])
        y[i] = rand(dist)
    end

    #define IHTVariable needed to run L0_reg
    v = IHTVariables(x, z, y, 1, k)
    
    #benchmark the result and return
    b = @benchmarkable L0_poisson_reg($v, $x, $z, $y, 1, $k, glm = "poisson") seconds=3000 samples=10
    return run(b)
end


#set random seed
srand(2018) 

#how many different sample sizes to run?
num_samples = 10

#problem dimension?
samples = [1000i for i in 1:num_samples]
p = 100000
k = 10
s = 0.1

#benchmark result on each sample size and store benchmark result
poisson_result = Vector{BenchmarkTools.Trial}(num_samples)
for i in 1:length(samples)
    poisson_result[i] = logistic_response(samples[i], p, k)
    println("completed " * string(i))
end