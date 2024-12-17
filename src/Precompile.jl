using Pkg
Pkg.activate(".")  # Activate the current project environment
Pkg.precompile()   # Precompile all packages
println("Precompilation complete!")