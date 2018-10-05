# scn-cpg

The implementation of [Structured Control Nets](https://arxiv.org/abs/1802.08311)

This is the shallow container repository and actual implementation can be found in my forked version of baselines: [coord-e/baselines](https://github.com/coord-e/baselines/tree/scn)

## Try

```
./setup.sh

pipenv run train humanoid-scn --num_timesteps 2e6 --alg ppo2 --network=mlp --num_hidden=16 --with_linear=True
# pipenv run train humanoid-cpg --num_timesteps 2e6 --alg ppo2 --network=mlp --num_hidden=16 --with_linear=True --observe_circular_ts=200 --without_network=True --with_cpg=True
```

Then you can plot the results:

```
pipenv run plot humanoid-scn
```

