
using Flux;
using Optimisers;
using ReinforcementLearning;
using ReinforcementLearningCore;
using ReinforcementLearningTrajectories;
using Distributions;

include("RL_Hohmann_Transfer_Env.jl");

function HT_Test()

    #declare Hohmann Transfer Environment object
    env = Hohmann_Transfer_Env()
    state_size = length( env.state );
    action_size = length( env.action );

    display( state_size );
    display( action_size );

    #check if the environment is runnable
    RLBase.test_runnable!( env )

    #check running with a random policy
    run(RandomPolicy(action_space(env)), env, StopAfterNEpisodes(1_000))

    #get reward per episode of random policy run
    hook = TotalRewardPerEpisode()

    #provide hook to get reward data from random policy
    run(RandomPolicy(action_space(env)), env, StopAfterNEpisodes(1_000), hook)

    #plot rewards
    p1 = plot(hook.rewards, label="Total Reward per Episode");
    display(p1);

    #create a policy network
    policy_nn = Chain(
            Dense(6, 32, tanh),
            Dense(32, 32, tanh),
            Dense(32, 6, tanh),
            Dense(6, 2 )
        )

    #intialize the policy optimizer
    policy_optimizer = Flux.Optimisers.Adam(0.001);

    #initialize a flux approximater policy
    approximater_policy = FluxApproximator( policy_nn, policy_optimizer );

    #define the trajectory container
    #example definition from RL website: 
    #    t = Trajectory(Traces(a=Int[], b=Bool[]), BatchSampler(3), InsertSampleRatioControler(1.0, 3));

    #    But the RL reinforcement page states that circular SART array traces are the version that we 
    #    typically want for RL, and it provides a blank function to define it with:
    #=
                function (capacity, state_size, state_eltype, action_size, action_eltype, reward_eltype)
                    MultiplexTraces{SS}(CircularArrayBuffer{state_eltype}(state_size..., capacity + 1)) +
                    MultiplexTraces{AA′}(CircularArrayBuffer{action_eltype}(action_size..., capacity + 1)) +
                    Traces(
                        reward=CircularArrayBuffer{reward_eltype}(1, capacity),
                        terminal=CircularArrayBuffer{Bool}(1, capacity),
                    )
                end
    =#

    function CircularArraySARTSTraces(capacity, state_size, state_eltype, action_size, action_eltype, reward_eltype)
        MultiplexTraces{SS}(CircularArrayBuffer{state_eltype}(state_size..., capacity + 1)) +
        MultiplexTraces{AA′}(CircularArrayBuffer{action_eltype}(action_size..., capacity + 1)) +
        Traces(
            reward=CircularArrayBuffer{reward_eltype}(1, capacity),
            terminal=CircularArrayBuffer{Bool}(1, capacity),
        )
    end

    CA_SARTS_Traces = CircularArraySARTSTraces( 1000, state_size, Float64, action_size, Float64, Float64 )
    #traj = Trajectory( CircularArraySARTSTraces( 1000, state_size, ::Float64, action_size, ::Float64, ::Float64 )

    #The RL julia documentation recommends wrapping a policy in an agent data structure. The "agent"
    #data structure is an abstract policy type, but it also takes in a policy as an argument, which
    #is confusing.
    #agent = Agent(  );
    



end



HT_Test();

