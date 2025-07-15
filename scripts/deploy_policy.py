import glob
import pickle as pkl
import lcm
import sys

from go2_gym_deploy.utils.deployment_runner import DeploymentRunner
from go2_gym_deploy.envs.lcm_agent import LCMAgent
from go2_gym_deploy.utils.cheetah_state_estimator import StateEstimator
from go2_gym_deploy.utils.command_profile import *
from genesis_lr.legged_gym.envs.go2.go2_config import GO2Cfg

import pathlib

# lcm多播通信的标准格式
lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

def load_and_run_policy(label, experiment_name, max_vel=1.0, max_yaw_vel=1.0):
    # load agent
    dirs = glob.glob(f"../../runs/{label}/*")
    logdir = sorted(dirs)[0]

# with open(logdir+"/parameters.pkl", 'rb') as file:
    # with open(logdir+"/parameters.pkl", 'rb') as file:
    #     pkl_cfg = pkl.load(file)
    #     print(pkl_cfg.keys())
    #     cfg = pkl_cfg["Cfg"]
    #     print(cfg.keys())

    # print('Config successfully loaded!')

    se = StateEstimator(lc) # controller에 대해서 세팅하는 부분 (조종기 input 및 sensor data(imu etc)에 대해서 data처리 하는 클래스. 좀 더 자세한 설명은 cheeta_state_estimator.py 참조)

    control_dt = 0.02
    command_profile = RCControllerProfile(dt=control_dt, state_estimator=se, x_scale=max_vel, y_scale=0.6, yaw_scale=max_yaw_vel)
    # RCControllerProfile -> 이 클래스의 경우에는 control input에 대해서 se에서 받은 raw data(라고 하기에는 조금의 처리가 들어가지만)를 통해서 실제로 처리하는 로직
    
    # -> 결론적으로 두 개의 클래스를 통해서 조종기의 input data 및 sensor data를 받고, 로봇에게 전달하기 위한 목표 command를 생성. 
    
    hardware_agent = LCMAgent(GO2Cfg, se, command_profile) # real world에서 얻은 data 및 command data를 처리하고, 강화학습을 실제로 수행했던 configuration 폴더를 들고와서 play.py를 real world에서 할 수 있도록 세팅.
    
    se.spin() # thread 설정 및 실행

    ###################################### 우리는 ppo_cse 사용하지 않기때문에 필요하지 않은 것으로 보임 ###############################
    
    # from go2_gym_deploy.envs.history_wrapper import HistoryWrapper
    # hardware_agent = HistoryWrapper(hardware_agent) # 이거 adaptation module 때문에 필요함.
    # print('Agent successfully created!')
    
    ################################################################################################################################
    
    policy = load_policy(logdir) # 여기에서 policy -> 아래의 load_policy함수에서 보면, obs를 받고 action을 출력하는 함수로 바뀌게 되는 상황임.
    
    print('Policy successfully loaded!')

    # load runner
    root = f"{pathlib.Path(__file__).parent.resolve()}/../../logs/"
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    deployment_runner = DeploymentRunner(experiment_name=experiment_name, se=None,
                                         log_root=f"{root}/{experiment_name}")
    deployment_runner.add_control_agent(hardware_agent, "hardware_closed_loop")
    deployment_runner.add_policy(policy)
    deployment_runner.add_command_profile(command_profile)

    if len(sys.argv) >= 2:
        max_steps = int(sys.argv[1])
    else:
        max_steps = 10000000
    print(f'max steps {max_steps}')

    deployment_runner.run(max_steps=max_steps, logging=True)

def load_policy(logdir):
    # try ------------------
    # body = torch.jit.load(logdir + '/checkpoints/body_latest.jit').to('cpu')
    
    
    ############## 원래 코드 ####################
    
    body = torch.jit.load(logdir + '/checkpoints/model_500_actor.jit')

    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/model_500_adaptor.jit').to('cpu')
    
    # actor_network = torch.jit.load("/home/kdg/basic/genesis_lr/logs/go2/exported/policies/deploy_policy.jit")

    def policy(obs, info):
        print(obs)
        
        latent = adaptation_module.forward(obs.to('cpu'))
        
        action = body.forward(torch.cat([obs.to('cpu'), latent], dim=-1))
        info['latent'] = latent
        
        return action

    return policy


if __name__ == '__main__':
    # label = "gait-conditioned-agility/pretrain-v0/train"
    label = "1"

    experiment_name = "example_experiment"

    # default:
    # max_vel=3.5, max_yaw_vel=5.0
    load_and_run_policy(label, experiment_name=experiment_name, max_vel=2.5, max_yaw_vel=5.0)
