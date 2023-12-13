# Experiments
In this file you can find the commands used to conduct the experiments reported in '*Peer Learning: Learning Complex Policies in Groups from Scratch via Action Recommendations*'
Where the job-id is created randomly by [wandb](https://wandb.ai) and is used as the seed for the corresponding experiment.

## Normal Setting
The MuJoCo experiments are rum with 4 peers, while the [Room](env/Room.md) environment is run with 2 peers.

### HalfCheetah
#### Peer Learning
```bash
run_peer.py --save-name <name> --job_id <id> --env HalfCheetah-v4 --env_args --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 10000 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 7.3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic True --use-agent-value True --use-trust True --peers-sample-with-noise True
```

#### Single Agent
```bash
run_peer.py --save-name <name> --job_id <id> --env HalfCheetah-v4 --env_args --agent-count 1 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 10000 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 7.3e-4" --agents_to_store 0 --T-decay 0 --mix-agents SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic False --use-agent-value False --use-trust False --peers-sample-with-noise True
```

#### Random Advice
```bash
run_peer.py --save-name <name> --job_id <id> --env HalfCheetah-v4 --env_args --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 10000 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 7.3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic False --use-agent-value False --use-trust False --peers-sample-with-noise True
```

#### Early Advice
```bash
run_peer.py --save-name <name> --job_id <id> --env HalfCheetah-v4 --env_args --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 10000 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 1 --learning_rate "lambda x: 7.3e-4" "lambda x: 7.3e-4" "lambda x: 7.3e-4" "lambda x: 7.3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 0 --sample-from-suggestions False --use-critic False --use-agent-value False --use-trust False --peers-sample-with-noise True --only-follow-peers True --max-peer-epochs 74
```

### Ant
#### Peer Learning

```bash
run_peer.py --save-name <name> --job_id <id> --env Ant-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic True --use-agent-value True --use-trust True --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

#### Single Agent
```bash
run_run.py --save-name <name> --job_id <id> --env Ant-v4 --env_args terminate_when_unhealthy=False --agent-count 1 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 --T-decay 0 --mix-agents SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic False --use-agent-value False --use-trust False --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

#### Random Advice
```bash
run_peer.py --save-name <name> --job_id <id> --env Ant-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic False --use-agent-value False --use-trust False --peers-sample-with-noise True
```

#### Early Advice
```bash
run_peer.py --save-name <name> --job_id <id> --env Ant-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 1 --learning_rate "lambda x: 7.3e-4" "lambda x: 7.3e-4" "lambda x: 7.3e-4" "lambda x: 7.3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 0 --sample-from-suggestions False --use-critic False --use-agent-value False --use-trust False --peers-sample-with-noise True --only-follow-peers True --max-peer-epochs 74
```

### Hopper
#### Peer Learning
```bash
run_peer.py --save-name <name> --job_id <id> --env Hopper-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic True --use-agent-value True --use-trust True --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

#### Single Agent
```bash
run_run.py --save-name <name> --job_id <id> --env Hopper-v4 --env_args terminate_when_unhealthy=False --agent-count 1 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 --T-decay 0 --mix-agents SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic False --use-agent-value False --use-trust False --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

#### Random Advice
```bash
run_peer.py --save-name <name> --job_id <id> --env Hopper-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic False --use-agent-value False --use-trust False --peers-sample-with-noise True
```

#### Early Advice
```bash
run_peer.py --save-name <name> --job_id <id> --env Hopper-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 1 --learning_rate "lambda x: 3e-4" "lambda x: 3e-4" "lambda x: 3e-4" "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 0 --sample-from-suggestions False --use-critic False --use-agent-value False --use-trust False --peers-sample-with-noise True --only-follow-peers True --max-peer-epochs 74
```

### Walker2d
#### Peer Learning
```bash
run_peer.py --save-name <name> --job_id <id> --env Walker2d-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic True --use-agent-value True --use-trust True --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 100000000
```

#### Single Agent
```bash
run_peer.py --save-name <name> --job_id <id> --env Walker2d-v4 --env_args terminate_when_unhealthy=False --agent-count 1 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 --T-decay 0 --mix-agents SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic False --use-agent-value False --use-trust False --peers-sample-with-noise True
```

#### Random Advice
```bash
run_peer.py --save-name <name> --job_id <id> --env Walker2d-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic False --use-agent-value False --use-trust False --peers-sample-with-noise True
```

#### Early Advice
```bash
run_peer.py --save-name <name> --job_id <id> --env Walker2d-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 1 --learning_rate "lambda x: 3e-4" "lambda x: 3e-4" "lambda x: 3e-4" "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 0 --sample-from-suggestions False --use-critic False --use-agent-value False --use-trust False --peers-sample-with-noise True --only-follow-peers True --max-peer-epochs 74
```

### Room
#### Peer Learning
```bash
run_peer.py --save-name <name> --job_id <id> --env Room-v21 --env_args --agent-count 2 --batch-size 256 --buffer-size 300000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 256 256 --follow-steps 10 --learning_rate "lambda x: 5e-4" --agents_to_store 0 --T-decay 0 --mix-agents DQN --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic True --use-agent-value True --use-trust True --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 100000000
```

#### Single Agent
```bash
run_peer.py --save-name <name> --job_id <id> --env Room-v21 --env_args --agent-count 1 --batch-size 256 --buffer-size 300000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 256 256 --follow-steps 10 --learning_rate "lambda x: 5e-4" --agents_to_store 0 --T-decay 0 --mix-agents DQN --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 0 --sample-from-suggestions False --use-critic False --use-agent-value False --use-trust False --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 100000000
```

#### Random Advice
```bash
run_peer.py --save-name <name> --job_id <id> --env Room-v21 --env_args --agent-count 2 --batch-size 256 --buffer-size 300000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 256 256 --follow-steps 10 --learning_rate "lambda x: 5e-4" --agents_to_store 0 --T-decay 0 --mix-agents DQN --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic False --use-agent-value False --use-trust False --peers-sample-with-noise True --only-follow-peers False
```

#### Early Advice
```bash
run_peer.py --save-name <name> --job_id <id> --env Room-v21 --env_args --agent-count 2 --batch-size 256 --buffer-size 300000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 256 256 --follow-steps 1 --learning_rate "lambda x: 5e-4" --agents_to_store 0 --T-decay 0 --mix-agents DQN --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 0 --min-epoch-length 0 --sample-from-suggestions False --use-critic False --use-agent-value False --use-trust False --peers-sample-with-noise True --only-follow-peers True --max-peer-epochs 5000
```

#### LeCTR
We used a modified (to work with our experiments) version of the code provided by the authors of the [paper](https://arxiv.org/pdf/1805.07830).

## Ablation Study
### HalfCheetah
#### With Advantage
##### Agent Values + Trust + Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env HalfCheetah-v4 --env_args --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 10000 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 7.3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic True --use-agent-value True --use-trust True --peers-sample-with-noise True
```

##### Agent Values + Trust
```bash
run_peer.py --save-name <name> --job_id <id> --env HalfCheetah-v4 --env_args --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 10000 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 7.3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic False --use-agent-value True --use-trust True --peers-sample-with-noise True
```

##### Agent Values + Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env HalfCheetah-v4 --env_args --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 10000 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 7.3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic True --use-agent-value True --use-trust False --peers-sample-with-noise True
```

##### Trust + Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env HalfCheetah-v4 --env_args --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 10000 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 7.3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic True --use-agent-value False --use-trust True --peers-sample-with-noise True
```

##### Agent Values
```bash
run_peer.py --save-name <name> --job_id <id> --env HalfCheetah-v4 --env_args --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 10000 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 7.3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic False --use-agent-value True --use-trust False --peers-sample-with-noise True
```

##### Trust
```bash
run_peer.py --save-name <name> --job_id <id> --env HalfCheetah-v4 --env_args --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 10000 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 7.3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic False --use-agent-value False --use-trust True --peers-sample-with-noise True
```

##### Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env HalfCheetah-v4 --env_args --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 10000 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 7.3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic True --use-agent-value False --use-trust False --peers-sample-with-noise True
```

#### Without Advantage
##### Agent Values + Trust + Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env HalfCheetah-v4 --env_args --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 10000 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 7.3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic True --use-agent-value True --use-trust True --peers-sample-with-noise True
```

##### Agent Values + Trust
```bash
run_peer.py --save-name <name> --job_id <id> --env HalfCheetah-v4 --env_args --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 10000 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 7.3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic False --use-agent-value True --use-trust True --peers-sample-with-noise True
```

##### Agent Values + Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env HalfCheetah-v4 --env_args --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 10000 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 7.3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic True --use-agent-value True --use-trust False --peers-sample-with-noise True
```

##### Trust + Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env HalfCheetah-v4 --env_args --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 10000 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 7.3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic True --use-agent-value False --use-trust True --peers-sample-with-noise True
```

##### Agent Values
```bash
run_peer.py --save-name <name> --job_id <id> --env HalfCheetah-v4 --env_args --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 10000 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 7.3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic False --use-agent-value True --use-trust False --peers-sample-with-noise True
```

##### Trust
```bash
run_peer.py --save-name <name> --job_id <id> --env HalfCheetah-v4 --env_args --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 10000 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 7.3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic False --use-agent-value False --use-trust True --peers-sample-with-noise True
```

##### Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env HalfCheetah-v4 --env_args --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 10000 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 7.3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic True --use-agent-value False --use-trust False --peers-sample-with-noise True
```

### Ant
#### With Advantage
##### Agent Values + Trust + Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env Ant-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic True --use-agent-value True --use-trust True --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

##### Agent Values + Trust
```bash
run_peer.py --save-name <name> --job_id <id> --env Ant-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic False --use-agent-value True --use-trust True --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

##### Agent Values + Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env Ant-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic True --use-agent-value True --use-trust False --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

##### Trust + Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env Ant-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic True --use-agent-value False --use-trust True --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

##### Agent Values
```bash
run_peer.py --save-name <name> --job_id <id> --env Ant-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic False --use-agent-value True --use-trust False --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

##### Trust
```bash
run_peer.py --save-name <name> --job_id <id> --env Ant-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic False --use-agent-value False --use-trust True --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

##### Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env Ant-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic True --use-agent-value False --use-trust False --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

#### Without Advantage
##### Agent Values + Trust + Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env Ant-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic True --use-agent-value True --use-trust True --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

##### Agent Values + Trust
```bash
run_peer.py --save-name <name> --job_id <id> --env Ant-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic False --use-agent-value True --use-trust True --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

##### Agent Values + Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env Ant-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic True --use-agent-value True --use-trust False --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

##### Trust + Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env Ant-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic True --use-agent-value False --use-trust True --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

##### Agent Values
```bash
run_peer.py --save-name <name> --job_id <id> --env Ant-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic False --use-agent-value True --use-trust False --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

##### Trust
```bash
run_peer.py --save-name <name> --job_id <id> --env Ant-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic False --use-agent-value False --use-trust True --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

##### Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env Ant-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic True --use-agent-value False --use-trust False --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

### Hopper
#### With Advantage
##### Agent Values + Trust + Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env Hopper-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic True --use-agent-value True --use-trust True --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

##### Agent Values + Trust
```bash
run_peer.py --save-name <name> --job_id <id> --env Hopper-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic False --use-agent-value True --use-trust True --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

##### Agent Values + Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env Hopper-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic True --use-agent-value True --use-trust False --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

##### Trust + Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env Hopper-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic True --use-agent-value False --use-trust True --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

##### Agent Values
```bash
run_peer.py --save-name <name> --job_id <id> --env Hopper-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic False --use-agent-value True --use-trust False --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

##### Trust
```bash
run_peer.py --save-name <name> --job_id <id> --env Hopper-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic False --use-agent-value False --use-trust True --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

##### Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env Hopper-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic True --use-agent-value False --use-trust False --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

#### Without Advantage
##### Agent Values + Trust + Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env Hopper-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic True --use-agent-value True --use-trust True --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

##### Agent Values + Trust
```bash
run_peer.py --save-name <name> --job_id <id> --env Hopper-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic False --use-agent-value True --use-trust True --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

##### Agent Values + Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env Hopper-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic True --use-agent-value True --use-trust False --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

##### Trust + Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env Hopper-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic True --use-agent-value False --use-trust True --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

##### Agent Values
```bash
run_peer.py --save-name <name> --job_id <id> --env Hopper-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic False --use-agent-value True --use-trust False --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

##### Trust
```bash
run_peer.py --save-name <name> --job_id <id> --env Hopper-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic False --use-agent-value False --use-trust True --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

##### Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env Hopper-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic True --use-agent-value False --use-trust False --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 10000000
```

### Walker2d
#### With Advantage
##### Agent Values + Trust + Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env Walker2d-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic True --use-agent-value True --use-trust True --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 100000000
```

##### Agent Values + Trust
```bash
run_peer.py --save-name <name> --job_id <id> --env Walker2d-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic False --use-agent-value True --use-trust True --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 100000000
```

##### Agent Values + Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env Walker2d-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic True --use-agent-value True --use-trust False --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 100000000
```

##### Trust + Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env Walker2d-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic True --use-agent-value False --use-trust True --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 100000000
```

##### Agent Values
```bash
run_peer.py --save-name <name> --job_id <id> --env Walker2d-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic False --use-agent-value True --use-trust False --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 100000000
```

##### Trust
```bash
run_peer.py --save-name <name> --job_id <id> --env Walker2d-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic False --use-agent-value False --use-trust True --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 100000000
```

##### Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env Walker2d-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic True --use-agent-value False --use-trust False --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 100000000
```

#### Without Advantage
```bash
run_peer.py --save-name <name> --job_id <id> --env Walker2d-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic True --use-agent-value True --use-trust True --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 100000000
```

##### Agent Values + Trust
```bash
run_peer.py --save-name <name> --job_id <id> --env Walker2d-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic False --use-agent-value True --use-trust True --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 100000000
```

##### Agent Values + Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env Walker2d-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic True --use-agent-value True --use-trust False --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 100000000
```

##### Trust + Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env Walker2d-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic True --use-agent-value False --use-trust True --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 100000000
```

##### Agent Values
```bash
run_peer.py --save-name <name> --job_id <id> --env Walker2d-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic False --use-agent-value True --use-trust False --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 100000000
```

##### Trust
```bash
run_peer.py --save-name <name> --job_id <id> --env Walker2d-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic False --use-agent-value False --use-trust True --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 100000000
```

##### Critic
```bash
run_peer.py --save-name <name> --job_id <id> --env Walker2d-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 1 2 3 --T-decay 0 --mix-agents SAC SAC SAC SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic True --use-agent-value False --use-trust False --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 100000000
```

## Adversarial Setting

All experiments have been conducted on the Room-v21 environment.

### Peer Learning

```bash
run_peer.py --save-name <name> --job_id <id> --env Room-v21 --env_args --agent-count 3 --batch-size 256 --buffer-size 300000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 256 256 --follow-steps 10 --learning_rate "lambda x: 5e-4" --agents_to_store 0 --T-decay 0 --mix-agents DQN DQN Adversarial --switch-ratio 0 --use-advantage True --epsilon 0.2 --T 1 --min-epoch-length 10000 --sample-from-suggestions True --use-critic True --use-agent-value True --use-trust True --peers-sample-with-noise True --only-follow-peers False
```

### Single Agent

```bash
run_peer.py --save-name <name> --job_id <id> --env Room-v21 --env_args --agent-count 1 --batch-size 256 --buffer-size 300000 --steps 1000000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 256 256 --follow-steps 10 --learning_rate "lambda x: 5e-4" --agents_to_store 0 --T-decay 0 --mix-agents DQN --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 0 --sample-from-suggestions False --use-critic False --use-agent-value False --use-trust False --peers-sample-with-noise True --only-follow-peers False --max-peer-epochs 100000000
```

## Number of Agents

All experiments have been conducted on the MuJoCo Hopper-v4 environment.

### 2 Agents

```bash
run_peer.py --save-name <name> --job_id <id> --env Hopper-v4 --env_args terminate_when_unhealthy=False --agent-count 2 --batch-size 256 --buffer-size 1000000 --steps 600000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 --T-decay 0 --mix-agents SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic True --use-agent-value True --use-trust True --peers-sample-with-noise True
```

### 4 Agents

```bash
run_peer.py --save-name <name> --job_id <id> --env Hopper-v4 --env_args terminate_when_unhealthy=False --agent-count 4 --batch-size 256 --buffer-size 1000000 --steps 600000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 --T-decay 0 --mix-agents SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic True --use-agent-value True --use-trust True --peers-sample-with-noise True
```

### 8 Agents

```bash
run_peer.py --save-name <name> --job_id <id> --env Hopper-v4 --env_args terminate_when_unhealthy=False --agent-count 8 --batch-size 256 --buffer-size 1000000 --steps 600000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <seed> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 --T-decay 0 --mix-agents SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic True --use-agent-value True --use-trust True --peers-sample-with-noise True
```

### 10 Agents

```bash
run_peer.py --save-name <name> --job_id <id> --env Hopper-v4 --env_args terminate_when_unhealthy=False --agent-count 10 --batch-size 256 --buffer-size 1000000 --steps 600000 --buffer-start-size 100 --gamma 0.99 --gradient_steps 8 --tau 0.02 --train_freq 8 --seed <id> --net-arch 150 200 --follow-steps 10 --learning_rate "lambda x: 3e-4" --agents_to_store 0 --T-decay 0 --mix-agents SAC --switch-ratio 0 --use-advantage False --epsilon 0.2 --T 1 --sample-from-suggestions True --use-critic True --use-agent-value True --use-trust True --peers-sample-with-noise True
```

