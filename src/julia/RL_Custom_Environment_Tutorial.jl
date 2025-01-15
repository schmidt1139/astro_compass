using ReinforcementLearning;
using Plots;

#necessary components/minimal interfaces
#=
action_space(env::YourEnv)
state(env::YourEnv)
state_space(env::YourEnv)
reward(env::YourEnv)
is_terminated(env::YourEnv)
reset!(env::YourEnv)
act!(env::YourEnv, action)
=#


Base.@kwdef mutable struct LotteryEnv <: AbstractEnv
    reward::Union{Nothing, Int} = nothing
end

struct LotteryAction{a}
    function LotteryAction(a)
        new{a}()
    end
end

#define action space
RLBase.action_space(env::LotteryEnv) = LotteryAction.([:PowerRich, :MegaHaul, nothing]);
RLBase.reward(env::LotteryEnv) = env.reward;
RLBase.state(env::LotteryEnv, ::Observation, ::DefaultPlayer) = !isnothing(env.reward);
RLBase.state_space(env::LotteryEnv) = [false, true];
RLBase.is_terminated(env::LotteryEnv) = !isnothing(env.reward);
RLBase.reset!(env::LotteryEnv) = env.reward = nothing;

function RLBase.act!(x::LotteryEnv, action)
    if action == LotteryAction(:PowerRich)
        x.reward = rand() < 0.01 ? 100_000_000 : -10
    elseif action == LotteryAction(:MegaHaul)
        x.reward = rand() < 0.05 ? 1_000_000 : -10
    elseif action == LotteryAction(nothing)
        x.reward = 0
    else
        @error "unknown action of $action"
    end
end

function test_environment()

    env = LotteryEnv();

    RLBase.test_runnable!( env );


    run(RandomPolicy(action_space(env)), env, StopAfterNEpisodes(1_000))
    EmptyHook()

    hook = TotalRewardPerEpisode()
    run(RandomPolicy(action_space(env)), env, StopAfterNEpisodes(1_000), hook)

    p1 = plot(hook.rewards);
    display(p1);

    p = QBasedPolicy(
           learner = TDLearner(
               TabularQApproximator(
                   n_state = length(state_space(env)),
                   n_action = length(action_space(env)),
               ), :SARS
           ),
           explorer = EpsilonGreedyExplorer(0.1)
       )

    wrapped_env = ActionTransformedEnv(
        StateTransformedEnv(
            env;
            state_mapping=s -> s ? 1 : 2,
            state_space_mapping = _ -> Base.OneTo(2)
        );
        action_mapping = i -> action_space(env)[i],
        action_space_mapping = _ -> Base.OneTo(3),
    )

    plan!(p, wrapped_env)

    h = TotalRewardPerEpisode()
    run(p, wrapped_env, StopAfterNEpisodes(1_000), h)

    p2 = plot(h.rewards)
    display(p2);
    
end

test_environment();












