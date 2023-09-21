# Get number of actions from gym action space

policy_net = DQN(nn_inputs, screen_height, screen_width, n_actions).to(device)
target_net = DQN(nn_inputs, screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

if LOAD_MODEL == True:
    policy_net_checkpoint = torch.load(
        "save_model/model_dict.pt"
    )  # best 3 is the default best
    target_net_checkpoint = torch.load("reslts/model_dict.pt")
    policy_net.load_state_dict(policy_net_checkpoint)
    target_net.load_state_dict(target_net_checkpoint)
    policy_net.eval()
    target_net.eval()
    stop_training = True  # if we want to load, then we don't train the network anymore
