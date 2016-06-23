using Base.Profile
n = 100
function f(x)
    A = reshape(x[1:n^2], n, n)
    B = reshape(x[n^2 + 1:2n^2], n, n)
    c = x[2n^2+1:end]
    trace(log(A * B .+ c))
end

@show N = 2n^2 + n
x = rand(N)

include("prototype6.jl")

∇f = grad(f, N)

stor = similar(x)
∇f(x, stor)

Profile.clear_malloc_data()
∇f(x, stor)
