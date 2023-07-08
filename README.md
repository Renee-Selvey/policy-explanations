# Policy explanations
Repository to compute explanations for a given policy, PDDL domain and problem.

## Docker installation
*Note: Include your Gurobi licence into this repository in order to use Gurobi
in a Docker container.*

To build the Docker image:
```
docker build -t asnets-bionic .
```

To run the container:
```
docker run -i --rm --mount type=bind,source=<full_path_to_shared_dir>,\
target=/home/asnets_user/shared -t asnets-bionic /bin/bash
```

Then, inside the Docker container, this repository called `home/shared`. 

Use `chmod` outside of the docker container on this repository to give write
permission to the docker container. This will allow for the results to be saved
as log files.

## Running manually
Running the command:
```
python compute_explanation.py weights.pkl domain.pddl problem.pddl \
--type sub_min_exp --path ./results
```

will compute a subset-minimal explanation for the given policy, where:
 - `weights.pkl` is the ASNets policy
 - `domain.pddl` is the PDDL domain
 - `problem.pddl` is the PDDL problem

The other optional arguments seen in `python compute_explanation.py --help` are
for previous iterations of the algorithm, which supported probabilistic domains
and the inclusion of auxilliary data in the ASNets.

## Running the experiments
The scripts in `./experiments` compute subset-minimal explanations for many
problems in a given domain. For instance, `blocks_scripts.py` finds minimal
explanations for the policies of blocksworld problems found in
`./experiments/experiment_problems/blocks`. Results of these experiments are
stored in `./results` in their respective domain's directory.