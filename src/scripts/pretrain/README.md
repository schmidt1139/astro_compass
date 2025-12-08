## Refactor Overview

I'm interested in a decoupled pipeline for rapid prototyping. 

Rather than have one script that does everything (generating buffer, pre-training, training, evaluating), instead have a individual files that do each part. 

This will allow for us to generate arbitrarily many replay buffers that can be saved and reused later. We can attempt different pre-training strategies and compare them head to head. We can also then decide if we want to include or exclude the ephem replay buffer on a pre-trained or un-trained RL agent. 


## Future Experiments
- Replay Buffers (Permute IC + total data)
    - Fixed IC + Random Target
    - Random IC + Random Target

    - Simple-lite -- 5_000 circular transfers
    - Simple -- 1_000_000 circular transfers
    - Complex-lite -- 5_000 elliptic transfers
    - Complex -- 1_000_000 elliptic transfers


- Pre-Training 
    - No Pre-training
    - See replay buffers above (x4)

- Agent Training
    - Initialize Replay Buffer (Y/N) # Prepopulate with ephems again or no
    - Save reward curves + compare 


- Evaluate Agent
    - Testing on same type of orbits (simple, complex)
    - Testing on new types of orbits


## Observations / Concerns
- Format of the scripts (pre-training + RL + eval all tightly coupled) -- fixed
- Configs should be toml for easier reading. --- fixed
- Plotting scripts are challenging to navigate --- fixed 
- GPU is not getting registered -- fixed
- Batch sizes are too small to meaningfully leverage GPU -- Not true
- Pre-training isn't saving the best models. -- Done
- The SAC data buffer size is 4_000_000 by default but we never reach this size or get rid of bad old data! 

## Questions
- What's the file format of the training-TBR-circular-001.tgz file? I can't unpack it with tar. --- probably corrupt file,  will be fixed
- Why is gamma 0.5? I assume this isn't discount, but something hamltonian controller -- true



## Priorities
- Get RL training to work as well as I can without pre-training


## December 6th

My priorities are to get a set of minimum viable experiments ready for the paper. I'm going to focus my attention on the TBR problem. 

Ideally the set of experiments are (Number of Training Datasets [M] ) x (Number of Environments [N]) x (Number of Environments - 1)

N(N-1) is because we could test the simpler environments after having been trained on the datasets for the more advanced environment pre-training data.

I'm proposing the following: 

Environments
---

(A) Earth / Mars

1) Fixed IC, Fixed Target, Fixed TOF (Easy)
2) Fixed IC, Random Target, Fixed TOF (Medium)
3) Random IC, Random Target, Fixed TOF (Hard)

4) (1) Repeat: Variable TOF
5) (2) Repeat: Variable TOF
6) (3) Repeat: Variable TOF

(B) Inner Planets

1) - 6) Same as above 

(C) All Planets

1) - 6) Same as above 

Pre-Training
---

1) No-Data --- None
2) Small-Data --- 100 data
3) Medium-Data --- 10,000 data
4) Large-Data --- 1,000,000 data



Totals
---
4 x 18 x 17 = 1200 Experiments 
Experiment takes ~ 1 hour = 50 days

Therefore, we are going to reduce the number of environments down to A1-A6
4 x 6 x 5 = 120 experiments = 5 days


## Considerations
- I'm not actually sure how much training data was generated and of what type
- It's possible that data is biased in certain ways based on the fact that some optimization was prematurely truncated (e.g. long transfers possibly excluded)
- It's possible that we can generate more trajectories if needed using tools like ASSET rather than python 


## Goals
- Update scripts to accept command line arguments
- Write a runner script that:
    - Generates a buffer if it doesn't already exist
    - Pre-trains the agent based on the requested data buffer
    - Trains the RL part of the agent
    - Runs the evaluation script
    - Dumps data to a file
- Develop evaluation metrics for the TBR problem
- Update the evaluate agent script to use / report these metrics

## Progress
- I want it to be easy to modify small parts of the environment / training setup without having to dig through tons of parameters. 
- This likely requires that we have a set of modular base config files that can be composed together 
- This will help minimize the amount of code duplication across all of the current / past config .txt and .toml files. 

- There is some shinanigans going on behind the scenes when initializaing environments
- There are the configs, the pulling of the config to pass to the class, and then defaults of the class not direclty in the constructor (?)
- I went ahead and changed it so that the environment params are in the config, the config gets passed directly to the constructor, overwritting any defaults that are explicit. 

- I'm not sure what to do with the paths yet. 
    - model path -> model config
    - replay buffer path -> environment config (?) 
        - that's what produces the buffer, but it's also what might be used for pre-training, which is ultimately for the model so model_config ?
    - plots -> ??
- Possibly should have a logging config file 



### December 8th
I think my goal is to finish the TBR refactor that allows for trivial experiment running + logging. 

I'm going to have a set of core toml files that always get loaded, and then a set of experiment toml files that the user can overwrite the existing toml variables. 

Priorities:
- Ensure the project has a common root directory that can always be used to organize folders in a consistent location
- Build lightweight regression tests for the new pre-training directory scripts to ensure future modifications don't change performance 
    - Perform for train_agent
    - Perform for pre-train-agent
        - Need to generate the buffer to complete pre-training
- Organize the configuration toml files into multiple files for different parts of the pipeline (common, model, env, pretraining, training). Include an experiments folder which can be used to overwrite certain parts of the core toml files, and write a config_util.py helper function for loading + overwritting.  