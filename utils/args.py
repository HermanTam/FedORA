import os
import argparse


def args_to_string(args):
    """
    Transform experiment's arguments into a string
    :param args:
    :return: string
    """
    if args.decentralized:
        return f"{args.experiment}_decentralized"

    args_string = ""

    args_to_show = ["experiment", "method"]
    for arg in args_to_show:
        args_string = os.path.join(args_string, str(getattr(args, arg)))

    if args.locally_tune_clients:
        args_string += "_adapt"

    return args_string


def parse_args(args_list=None):
    parser = argparse.ArgumentParser(description=__doc__)
    # Core
    parser.add_argument('experiment', type=str, help='name of experiment')
    parser.add_argument('method', type=str, help='method: FedAvg|FedEM|local|FedProx|L2SGD|pFedMe|AFL|FFL|clustered')
    parser.add_argument('--decentralized', action='store_true', help='use decentralized version (D-SGD / D-EM)')
    parser.add_argument('--sampling_rate', type=float, default=1.0, help='proportion of clients per round')
    parser.add_argument('--input_dimension', type=int, default=None, help='synthetic only')
    parser.add_argument('--output_dimension', type=int, default=None, help='synthetic only')
    parser.add_argument('--n_learners', type=int, default=3, help='number of learners (FedEM)')
    parser.add_argument('--n_rounds', type=int, default=1, help='communication rounds')
    parser.add_argument('--bz', type=int, default=1, help='batch size')
    parser.add_argument('--local_steps', type=int, default=1, help='local steps per round')
    parser.add_argument('--log_freq', type=int, default=1, help='global log frequency (rounds)')
    parser.add_argument('--device', type=int, default=0, help='GPU index (0=cuda:0, 1=cuda:1). For CPU, set CUDA_VISIBLE_DEVICES=""')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_lambda', type=float, default=0., help='client-weight lr (agnostic FL)')
    parser.add_argument('--lr_scheduler', type=str, default='constant', help='sqrt|linear|cosine_annealing|multi_step|constant')
    parser.add_argument('--mu', type=float, default=0, help='prox/penalty weight')
    parser.add_argument('--gamma', type=float, default=0.05, help='tolerance threshold (FedConceptEM)')
    parser.add_argument('--suffix', type=str, default='', help='suffix for log files')
    parser.add_argument('--communication_probability', type=float, default=0.1, help='L2SGD only')
    parser.add_argument('--q', type=float, default=1., help='FFL fairness hyper-parameter')
    parser.add_argument('--locally_tune_clients', action='store_true', help='tune clients locally for 1 epoch before logs')
    parser.add_argument('--split', action='store_true', help='split features and classifiers')
    parser.add_argument('--hard_cluster', action='store_true', help='use hard cluster for prediction')
    parser.add_argument('--binary', action='store_true', help='binary classification loss')
    parser.add_argument('--domain_disc', action='store_true', help='add domain discriminator')
    parser.add_argument('--diagnosis_mode', type=str, choices=['binary', 'multiclass'], default='binary', help='drift diagnosis granularity')
    parser.add_argument('--objective_aware', action='store_true', help='enable objective-aware routing (uses client.objective if set)')
    parser.add_argument('--objective_assignment', type=str, default=None,
                        help='OPTIONAL: assign objectives: first_half:G,second_half:P | all:G | all:P | random_50_50')

    # Adaptive thresholds
    parser.add_argument('--adaptive_thresholds', action='store_true', help='use adaptive drift thresholds (mu+tau*sigma)')
    parser.add_argument('--threshold_tau', type=float, default=3.0, help='tau for adaptive thresholds')
    parser.add_argument('--threshold_window', type=int, default=10, help='baseline rolling window')
    parser.add_argument('--use_statistical_tests', action='store_true', help='use chi-square & KS tests instead of mu+tau*sigma')

    # Differential Privacy
    parser.add_argument('--use_dp', action='store_true', help='enable differential privacy for prototypes')
    parser.add_argument('--dp_epsilon', type=float, default=1.0, help='DP epsilon')
    parser.add_argument('--dp_delta', type=float, default=1e-5, help='DP delta')
    parser.add_argument('--dp_mechanism', type=str, choices=['gaussian', 'laplace'], default='gaussian', help='DP mechanism')
    parser.add_argument('--dp_max_norm', type=float, default=10.0, help='DP clipping bound')

    # Global eval & ablations
    parser.add_argument('--global_eval', action='store_true', default=True, help='evaluate on fixed 0° global test set (default ON)')
    parser.add_argument('--abl_all_reset', action='store_true', help='force reset for non-real drifts (real still resets)')
    parser.add_argument('--abl_all_merge', action='store_true', help='force merge for non-real drifts (real still resets)')
    parser.add_argument('--eval_all_past_concepts', action='store_true', help='at final slot, evaluate over all past per-slot test iterators')

    # Merge policy (routing)
    parser.add_argument('--merge_policy', type=str,
                        default='real:allP,label:halfP,halfG,feature:halfP,halfG,none:halfP,halfG',
                        help=('Routing policy per drift type (continual vs reset). Examples: '
                              '"real:allP,label:halfP,halfG,feature:allG,none:halfP,halfG" or '
                              '"real:reset;label:follow;feature:follow;none:follow". '
                              'Aliases: reset=allP (reset/no merge), merge=allG (merge), split=halfP,halfG (50/50 or objective-aware), '
                              'follow = objective-aware (P=reset, G=merge). NOTE: quote the whole string if you use semicolons.'))

    # Misc
    parser.add_argument('--phi_model', action='store_true', help='add auxiliary phi model (stoCFL)')
    parser.add_argument('--validation', action='store_true', help='use validation instead of test')
    parser.add_argument('--verbose', type=int, default=0, help='verbosity 0|1|2')
    parser.add_argument('--logs_dir', default=argparse.SUPPRESS, help='logs root (auto if omitted)')
    parser.add_argument('--save_dir', default=argparse.SUPPRESS, help='directory to save checkpoints at end')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--seeds', type=str, default=None, help='comma-separated seeds (overrides --seed)')
    parser.add_argument('--embedding_dimension', type=int, default=32, help='internal embedding size')
    parser.add_argument('--em_step', type=int, default=1, help='EM update interval (steps)')
    parser.add_argument('--n_gmm', type=int, default=1, help='number of GMM components')
    parser.add_argument('--T', type=int, default=2, help='number of time slots (concept drift)')

    args = parser.parse_args(args_list) if args_list else parser.parse_args()
    return args
