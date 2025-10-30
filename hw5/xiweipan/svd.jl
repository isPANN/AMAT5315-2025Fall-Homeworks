include("../download_mnist.jl")
using LinearAlgebra
using Images
using CairoMakie
using Statistics

train_images, train_labels = download_mnist(:train)
@show size(train_images)

X = reshape(train_images, 28*28, :)
U, S, V = svd(X)
@show S

k_values = [10, 50, 100, 200]


function compression_ratio(k, m, n)
    return (m * n) / (k * (m + n + 1))
end

function calculate_mse(original, reconstructed)
    return mean((original .- reconstructed).^2)
end

function calculate_frobenius_error(original, reconstructed)
    return norm(original - reconstructed)
end

crs = Float64[]
mse_errors = Float64[]
frobenius_errors = Float64[]

for k in k_values
    X_compressed = U[:, 1:k] * diagm(S[1:k]) * V[:, 1:k]'
    images = reshape(X_compressed, 28, 28, :)
    @assert size(images) == (28, 28, 60000)
    push!(crs, compression_ratio(k, 28*28, 60000))
    push!(mse_errors, calculate_mse(X, X_compressed))
    push!(frobenius_errors, calculate_frobenius_error(X, X_compressed))
    
end

fig = Figure()
CairoMakie.Axis(fig[1, 1], xlabel = "k", ylabel = "compression ratio")
scatterlines!(k_values, crs, color = :red)
CairoMakie.Axis(fig[1, 2], xlabel = "k", ylabel = "mse error")
scatterlines!(k_values, mse_errors, color = :blue)
fig
save("svd_compression_ratio.png", fig)