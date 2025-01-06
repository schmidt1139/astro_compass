using Pkg
Pkg.activate(".")  # Activate the current project environment
Pkg.precompile()   # Precompile all packages
println("Precompilation complete!")


deps = Pkg.dependencies()

#display(deps)

#=
for (UUID, pkg_info) in deps

    flag_RL_dep = false;
    if ( occursin("ReinforcementLearning", string(pkg_info) ) )
        flag_RL_dep = true;
    end

    if ( flag_RL_dep == true )
        println("UUID: " * string(UUID) );
        println("Pkg info: " * string(pkg_info) );
        println("Flag RL dep: " * string(flag_RL_dep) );
        println("\n");
    end

end
=#
