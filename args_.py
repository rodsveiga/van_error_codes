import argparse

parser = argparse.ArgumentParser()

group = parser.add_argument_group('physics parameters')


group.add_argument(
    '--N',
    type=int,
    default=16,
    help='original message length')


group.add_argument(
    '--M',
    type=int,
    default=32,
    help='codeword length')


group.add_argument(
    '--K',
    type=int,
    default=4,
    help='M sets of K randomly chosen indices')

group.add_argument(
    '--num_codes',
    type=int,
    default=1,
    help='Number of codes')



group.add_argument(
        '--p', 
        type=float, 
        default=0.5, 
        help='noise flip rate -- p = 1 / (1 + exp(2*beta))')


group.add_argument(
        '--p_prior', 
        type=float, 
        default=0.1, 
        help='prior flip rate -- p_prior = 1 / (1 + exp(2*beta_prior))')

group.add_argument('--tensorboard', action='store_true', help='tensorboard monitor')

group.add_argument('--save_model', action='store_true', help='save model and message')

group.add_argument('--log', action='store_true', help='save training info in a file')

group.add_argument('--laptop', action='store_true', help='running in the personal laptop')

group.add_argument('--BP', action='store_true', help='BP computation')

group.add_argument(
    '--BP_it',
    type=int,
    default=10,
    help='Number of BP iterations')



group = parser.add_argument_group('network parameters')


group.add_argument(
    '--net',
    type=str,
    default='made',
    choices=['made'],
    help='network type')

group.add_argument('--net_depth', type=int, default=3, help='network depth')
group.add_argument('--net_width', type=int, default=4, help='network width')

group.add_argument(
    '--half_kernel_size', type=int, default=1, help='(kernel_size - 1) // 2')
group.add_argument(
    '--dtype',
    type=str,
    default='float32',
    choices=['float32', 'float64'],
    help='dtype')
group.add_argument('--bias', action='store_true', help='use bias')

group.add_argument(
    '--z2', action='store_true', help='use Z2 symmetry in sample and loss')


group.add_argument(
    '--x_hat_clip',
    type=float,
    default=0,
    help='value to clip x_hat around 0 and 1, 0 for disabled')
group.add_argument(
    '--final_conv',
    action='store_true',
    help='add an additional conv layer before sigmoid')
group.add_argument(
    '--epsilon',
    type=float,
    default=1e-7,
    help='small number to avoid 0 in division and log')

group.add_argument('--res_block', action='store_true', help='use res block')



group = parser.add_argument_group('optimizer parameters')


group.add_argument(
    '--seed', type=int, default=0, help='random seed, 0 for randomized')

group.add_argument(
    '--optimizer',
    type=str,
    default='adam',
    choices=['sgd', 'sgdm', 'rmsprop', 'adam', 'adam0.5'],
    help='optimizer')

group.add_argument(
    '--num_messages', type=int, default=10**3, help='number of messages')

group.add_argument('--lr', type=float, default=1e-3, help='learning rate')

group.add_argument(
    '--max_step', type=int, default=10**4, help='maximum number of steps')

group.add_argument(
    '--lr_schedule', action='store_true', help='use learning rate scheduling')

group.add_argument(
    '--beta_anneal',
    type=float,
    default=0,
    help='speed to change beta from 0 to final value, 0 for disabled')

group.add_argument(
    '--clip_grad',
    type=float,
    default=0,
    help='global norm to clip gradients, 0 for disabled')



group = parser.add_argument_group('system parameters')


group.add_argument(
    '--no_stdout',
    action='store_true',
    help='do not print log to stdout, for better performance')



group.add_argument(
    '--monitor_ov', action='store_true', help='clear checkpoint')

group.add_argument(
    '--monitor_freq',
    type=int,
    default=1000,
    help='number of steps to monitor the overlap')


group.add_argument(
    '--monitor_num_samples',
    type=int,
    default=99,
    help='number of samples from the trained to monitor the overlap')


group.add_argument(
    '--clear_checkpoint', action='store_true', help='clear checkpoint')


group.add_argument(
    '--print_step',
    type=int,
    default=1,
    help='number of steps to print log, 0 for disabled')


group.add_argument(
    '--save_step',
    type=int,
    default=0,
    help='number of steps to save network weights, 0 for disabled')


group.add_argument(
    '--visual_step',
    type=int,
    default=100,
    help='number of steps to visualize samples, 0 for disabled')


group.add_argument(
    '--save_sample', action='store_true', help='save samples on print_step')


group.add_argument(
    '--print_sample',
    type=int,
    default=0,
    help='number of samples to print to log on visual_step, 0 for disabled')


group.add_argument(
    '--print_grad',
    action='store_true',
    help='print summary of gradients for each parameter on visual_step')


group.add_argument(
    '--cuda', type=int, default=-1, help='ID of GPU to use, -1 for disabled')


group.add_argument(
    '--out_infix',
    type=str,
    default='',
    help='infix in output filename to distinguish repeated runs')

group.add_argument(
    '-o',
    '--out_dir',
    type=str,
    default='out',
    help='directory prefix for output, empty for disabled')

args = parser.parse_args()
