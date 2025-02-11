mutable struct Celestial_Body
    name::String;               #Name of the body
    position::Array{Float64};   #position vector in km
    velocity::Array{Float64};   #velocity vector in km/s
    mu::Float64;                #gravitational parameter
    r::Float64;                 #body radius
end

