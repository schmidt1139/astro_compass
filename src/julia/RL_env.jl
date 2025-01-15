
include("Celestial_Body.jl");
include("Ephemeris.jl");
include("Spacecraft.jl");
include("Spacecraft_Propagation.jl");
include("Policy.jl");
include("Reward.jl");
include("Hohmann_Transfer_RL_Support_Functions.jl");


using Plots;
using ReinforcementLearning;
using ReinforcementLearningTrajectories;


function RL_Tutorial()
    
    println("\n\n\n\n\n\n\n\n\nReinforcement Learning Tutorial");

    env = RandomWalk1D()

    S = state_space(env);

    s = state(env);

    A = action_space(env);

    flag_terminated = is_terminated(env);

    while true

        act!(env, rand(A) )

        if (is_terminated(env) == true)
            break;
        end
        
    end


    s = state(env);
    r = reward(env);



    #---------------------------------------------------------------------------------------------
    #Run random walk policy
    display(
    run(
           RandomPolicy(),
           RandomWalk1D(),
           StopAfterNEpisodes(10),
           TotalRewardPerEpisode()
       )
    )

    #Establish Q based policy
    NS = length(S);
    NA = length(A);
    policy = QBasedPolicy(
           learner = TDLearner(
                   TabularQApproximator(
                       n_state = NS,
                       n_action = NA,
                   ),
                   :SARS
               ),
           explorer = EpsilonGreedyExplorer(0.1)
       )

    
    #Run the Q based policy
    display(
    run(
        policy,
        RandomWalk1D(),
        StopAfterNEpisodes(10),
        TotalRewardPerEpisode()
        )
    )

    trajectory = Trajectory(
           ElasticArraySARTSTraces(;
               state = Int64 => (),
               action = Int64 => (),
               reward = Float64 => (),
               terminal = Bool => (),
           ),
           DummySampler(),
           InsertSampleRatioController(),
       )

    
    agent = Agent(
        policy = RandomPolicy(),
        trajectory = trajectory
    )

    display( run(agent,env, StopAfterNEpisodes(10), TotalRewardPerEpisode() ) );

end

RL_Tutorial();