# Fast Online Learning of CLiFF-maps in Changing Environments

## Method
In this paper we propose an online update method of the CLiFF-map (an advanced map of dynamics type that models motion patterns as velocity and orientation mixtures) to actively detect and adapt to human flow changes. As new observations are collected, our goal is to update a CLiFF-map to effectively and accurately in- tegrate them, while retaining relevant historic motion patterns. The proposed online update method maintains a probabilistic representation in each observed location, updating parameters by continuously tracking sufficient statistics. In experiments using both synthetic and real-world datasets, we show that our method is able to maintain accurate representations of human motion dynamics, contributing to high performance flow-compliant planning downstream tasks, while being orders of magnitude faster than the comparable baselines.

## Dataset
Use a read world dataset, ATC dataset and a synthetic dataset (*den520d*, named MAPF in the code) for evaluation. In the experiments across both datasets, the grid resolution for MoD is set to 1m. For each condition in the *den520d* or each hour in the ATC, we randomly sample 10% of the data for testing and use the remaining 90% for training.

- ATC: in `dataset/atc`. The original ATC dataset is https://dil.atr.jp/crest2010_HRI/ATC_dataset. We downsample to 1Hz and split into each hour, from 9-10 to 20-21. First day of ATC dataset (Oct 24) is used in the experiment. 

- *den520d*: in `dataset/mapf`. This dataset is generated using a map from the Multi-Agent Path-Finding (MAPF) Benchmark <a href="#references">[1]</a>
 and features two distinct flow patterns: Condition A and Condition B (named *initial* and *update* in the code). We simulate a change in human flow from Condition A to Condition B, where the dominant flow in Condition B is reversed compared to that in Condition A. We discretize the den520d obstacle map with 1m cell resolution. In this map, randomized human trajectories are simulated using stochastic optimal control path finding, based on Markov decision processes <a href="#references">[2]</a>. In each condition 1000 trajectories are generated.

## Run
The configuration files are `config_mapf.yaml` and `config_atc.yaml`.

To run the online update method and more variations:
```bash
python3 main_atc.py --build-type <build_type> # for atc dataset
python3 main_mapf.py --build-type <build_type> # for den520d dataset
```
Here the `<build_type>` can be: `online`, `all` and `interval`.

- `online`: online update model
- `all`: a model built from all observations in iteratioin 1 to *k*. This model is named `history` in paper.
- `interval`: a model with observations only in ieration *k*.

The generated maps of dynamics will be placed in `cliffmaps/atc` and `cliffmaps/mapf`.

To plot the maps of dynamics, run
```bash
python3 plot_cliff_map.py
```
- Check code to plot cliffmaps for ATC / *den520d* datasets.

The generated figure will be placed in same dir as cliffmaps, like: `cliffmaps/atc/online/figs`.


## References
**[1]** R. Stern et al. “Multi-Agent Pathfinding: Definitions, Variants, and Benchmarks”. In: Symposium on Com- binatorial Search (SoCS) (2019), pp. 151–158.

**[2]** A. Rudenko, L. Palmieri, J. Doellinger, A. J. Lilien- thal, and K. O. Arras. “Learning Occupancy Priors of Human Motion From Semantic Maps of Urban Environments”. In: IEEE Robotics and Automation Letters 6.2 (2021), pp. 3248–3255.