import sys
from experiment.training import train

arm = sys.argv[1]
train(num_steps=500_000, seed=0, arm=arm, device="cuda", result="results2", resume=True, logstep=50_000, buffer_capacity=200_000)
