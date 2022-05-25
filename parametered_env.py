# Generic imports
import os
import numpy

from tensorforce.execution import Runner
from tensorforce import Environment, Runner
# Custom imports
from environment import *

from threading import Thread

from tensorforce import Environment, Runner

# Parameters not going into the environment
learning_frequency      = 40
batch_size              = learning_frequency
learning_rate           = 1.0e-3
gae_lambda              = 0.95
clipping_ratio          = 0.2
entropy                 = 0.01
model_dir               = '.'

# Define environment
def resume_env():
    # Environment parameters
    reset_dir               = 'reset/4'
    nb_pts_to_move          = 4
    pts_to_move             = [0,1,2,3]
    nb_ctrls_per_episode    = 1
    nb_episodes             = 50000
    max_deformation         = 3.0
    restart_from_cylinder   = True
    replace_shape           = True
    comp_dir                = '.'
    restore_model           = False
    saving_model_period     = 10
    cfl                     = 0.5
    reynolds                = 10.0
    output_vtu              = True
    shape_h                 = 11.0
    domain_h                = 0.7
    cell_limit              = 50000
    xmin                    =-15.0
    xmax                    = 30.0
    ymin                    =-15.0
    ymax                    = 15.0
    final_time              = 4*(xmax-xmin)

    # Define environment
    environment=env(nb_pts_to_move, pts_to_move,
                    nb_ctrls_per_episode, nb_episodes,
                    max_deformation,
                    restart_from_cylinder,
                    replace_shape,
                    comp_dir,
                    restore_model,
                    saving_model_period,
                    final_time, cfl, reynolds,
                    output_vtu,
                    shape_h, domain_h,
                    cell_limit,
                    reset_dir,
                    xmin, xmax, ymin, ymax)

    return(environment)




def main():
    environment_1= resume_env()
    print(type(environment_1))

    # Multi-actor runner, automatically if environment.num_actors() > 1
    agent = dict(
        agent='ppo',
        # Automatically configured network
        network='auto',
        # PPO optimization parameters
        batch_size=10, update_frequency=10, learning_rate=3e-4, multi_step=10,
        subsampling_fraction=0.33,
        # Reward estimation
        likelihood_ratio_clipping=0.2, discount=0.99, predict_terminal_values=False,
        reward_processing=None,
        # Baseline network and optimizer
        baseline=dict(type='auto', size=32, depth=1),
        baseline_optimizer=dict(optimizer='adam', learning_rate=1e-3, multi_step=10),
        # Regularization
        l2_regularization=0.0, entropy_regularization=0.01,
        # Preprocessing
        state_preprocessing='linear_normalization',
        # Exploration
        exploration=0.0, variable_noise=0.0,
        # Default additional config values
        config=None,
        # Save agent every 10 updates and keep the 5 most recent checkpoints
        saver=dict(directory='model', frequency=10, max_checkpoints=5),
        # Log all available Tensorboard summaries
        summarizer=dict(directory='summaries', summaries='all'),
        # Do not record agent-environment interaction trace
        recorder=None
    )

    runner = Runner(
        agent=agent,
        environment= resume_env(),
        max_episode_timesteps=environment_1.nb_ctrls_per_episode
        
    )
    
    runner.run(num_episodes=environment_1.nb_episodes)
    runner.close()



if __name__ == "__main__":
    main()
