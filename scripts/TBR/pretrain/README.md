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