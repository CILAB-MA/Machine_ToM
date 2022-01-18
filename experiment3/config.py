
def get_configs(num_exp, max_step=51):

    # get sub exp configs
    height = 11
    width = 11
    if num_exp == 1:
        num_past = 4
        num_step = max_step
        move_penalty = -0.01

    else:
        assert ('You put the wrong exp_num.')

    env_kwargs = dict(height=height, width=width, pixel_per_grid=8, preference=100, exp=2, save=True)
    model_kwargs = dict(num_past=num_past, num_input=11, num_exp=num_exp, num_step=num_step,
                        device='cuda')
    exp_kwargs = dict(num_past=num_past, num_step=num_step, move_penalty=move_penalty)
    agent_kwargs = 'deep_rl_agent'

    return exp_kwargs, env_kwargs, model_kwargs, agent_kwargs