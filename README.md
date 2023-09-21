# IN5490 Project repo

Project 22: AI Thinking Fast and Slow in a Multi-Agent environment

**Description and Objectives:**

In his best-seller “Thinking Fast and Slow”, Nobel Prize winner Daniel Kahneman proposes a thesis that states that there exist two modes of thought: System 1 which is fast, instinctive, and emotional, and System 2 which is slow, but also more deliberate and logical. This idea is now extensively used in practice, notably to help in athletes’ performance assessment [1]. The book however has also faced controversy, notably in light of the current replication crisis [2]. Regardless of the psychological merit of this theory in humans, however, it would be interesting to explore how designing an AI system using this paradigm could lead to both innovation and improvement in the field.

The main goal of the current project is thus to explore “computation-conscious” and “self-adjusting computation” algorithms within the context of reinforcement learning. An example of useful algorithms to consider is Early-Exist Neural Networks which can adapt their computation and thus inference-time based on how complex a given input is. A possible approach for the project is to consider a competitive multi-agents 3D environment using Unity’s ML-Agents and compare how different agents survived in this environment with and without the capacity to adapt their computational budget dynamically.

# Food Collector environment

This is a run of four of our agents interacting with the environment, rendered in Unity.


https://media.github.uio.no/user/3730/files/2f00ed9f-6272-4d50-b9c4-4bdbcb5ff615


If there is problem with video playback, the video is also available in the directory "video".

# How to run

## Cartpole

```
python3 DQN.py

Options:
    --fast_path
                If added, a fast_model will be generated. Model will be saved at path

    --slow_path
                If added, a slow_model will be generated. Model will be saved at path

    --eval
                If added, evaluation of the networks in cartpole will be run. Paths to pretrained models needs te be added

    --runs
                How many runs of evaluation to be run
```

## Unity Food Collector

```
python3 DQN_Unity.py

Options:
    --train
            If added we set the environment in train mode.
            If added we need to pass either --slow or --fast

            --slow
                    If added we train the ResNet-18

            --fast
                    If added we train the small CNN.
```

If you want to make a model, the hyper parameters can be found in the top of the file. All the network archectures can be found in DQN_model.py.

**References:**

1. Daniel, Kahneman. "Thinking, fast and slow." (2017).
2. Teerapittayanon, Surat, Bradley McDanel, and Hsiang-Tsung Kung. "Branchynet: Fast inference via early exiting from deep neural networks." 2016 23rd International Conference on Pattern Recognition (ICPR). IEEE, 2016.
3. Vinyals, Oriol, et al. "Grandmaster level in StarCraft II using multi-agent reinforcement learning." Nature 575.7782 (2019): 350-354.

```

```
