
include("Celestial_Body.jl");
include("Ephemeris.jl");
include("Spacecraft.jl");
include("Spacecraft_Propagation.jl");
include("Policy.jl");
include("Reward.jl");
include("Hohmann_Transfer_RL_Support_Functions.jl");


using Plots;
using ReinforcementLearning;

println("Test");

env = RandomWalk1D();