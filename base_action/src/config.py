# coding:utf-8
import argparse
import datetime
import os

def get_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()

parser.add_argument('--is_training', default=True)
parser.add_argument('--is_using_gpu', default=True)
parser.add_argument('--gpu_device', default=0)
#-----------------------------comet_ml Setting-----------------------------------#
parser.add_argument('--api_key', default = "Need to fill in")
parser.add_argument('--project_name', default = get_ip())
parser.add_argument('--workspace', default = "Need to fill in")
#-----------------------------Parameters Setting-----------------------------------#
parser.add_argument('--replaybuffer_size', type=int, default = 100000)
parser.add_argument('--state_size', type=int, default = 24*2+2)
parser.add_argument('--action_size', type=int, default = 2)
# parser.add_argument('--hidden_size', type=int, default = 512)
parser.add_argument('--hidden_size', type=int, default = 256)
parser.add_argument('--batch_size', type=int, default = 128)
parser.add_argument('--action_v_Min', type=float, default = 0.0) # m/s
parser.add_argument('--action_v_Max', type=float, default = 0.52) # m/s
parser.add_argument('--action_w_Min', type=float, default = -2.79) # rad/s
parser.add_argument('--action_w_Max', type=float, default = 2.79) # rad/s
parser.add_argument('--start_episodes', type=int, default = 0)
parser.add_argument('--max_episodes', type=int, default = 300000)
parser.add_argument('--max_steps', type=int, default = 500)
parser.add_argument('--actor_lr', type=float, default=3e-4,
                    help='actor learning rate')
parser.add_argument('--critic_lr', type=float, default=3e-4,
                    help='critic learning rate')
parser.add_argument('--alpha_lr', type=float, default=3e-4,
                    help='alpha learning rate')
parser.add_argument('--logdir', type=str, default='./logs',
                    help='tensorboardx logs directory')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for reward')
parser.add_argument('--tau', type=float, default=1e-2,
                    help='target smoothing coefficient(τ)')
#-----------------------------Model save Setting---------------------------------#
parser.add_argument('--data_date', default= datetime.datetime.now().strftime("%Y%m%d-%H-%M"))
parser.add_argument('--stage_num', default=1)
args = parser.parse_args()
parser.add_argument('--save_path', type=str, default = '{}/SAC_model/{}_stage{}/'.format(CODE_DIR, args.data_date, args.stage_num), help='')
# parser.add_argument('--save_path', type=str, default = '{}/SAC_model/{}social_stage{}/'.format(CODE_DIR, args.data_date, args.stage_num), help='')
parser.add_argument('--plot_env', default = u'{}_en{}'.format(args.data_date, args.stage_num))
# parser.add_argument('--plot_env', default = u'{}Social_en{}'.format(args.data_date, args.stage_num))
parser.add_argument('--plot_title', type=str, default = 'Environment {}'.format(args.stage_num))
parser.add_argument('--save_treshold', type=int, default=100)
#--------------------------------------------------------------------------------#
# parser.add_argument('--load_model', type=str, default=None)
# parser.add_argument('--save_path', default='./save_model/', help='')
# parser.add_argument('--max_iter_num', type=int, default=1000)
# parser.add_argument('--goal_score', type=int, default=-300)
args = parser.parse_args()
