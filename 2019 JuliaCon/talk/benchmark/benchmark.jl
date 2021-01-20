using SnpArrays
using BenchmarkTools
using LinearAlgebra
using MendelIHT
using Random

Random.seed!(1234)

n = 5000
p = 300_000
b = rand(Float64, p)
x = simulate_random_snparray(n, p, "compressed.bed")

X = SnpBitMatrix{Float64}(x, model=ADDITIVE_MODEL, center=true, scale=true); 
println("compressed timing")
@benchmark X * b seconds=120

X = convert(Matrix{Float64}, x, center=true, scale=true)
println("uncompressed timing")
@benchmark X * b seconds=120

rm("compressed.bed", force=true)