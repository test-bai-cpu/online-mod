import argparse
import sys

from mod_online_update import OnlineUpdateMoD
from mod_build_all import AllBuildMoD
from mod_build_interval import IntervalBuildMoD
from data_loader import DataLoader


def get_args():
    parser = argparse.ArgumentParser(description='group_simulator')
    parser.add_argument(
        "--build-type",
        type=str,
        default="online",
        help="The type of build to perform. Options: 'online', 'all', 'interval'. Default: 'online'."
    )

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
        
    return args

if __name__ == "__main__":
    args = get_args()
    
    if args.build_type == "online":
        buildMoD = OnlineUpdateMoD(
            config_file="config_atc.yaml",
            current_cliff=None,
            output_cliff_folder=f"cliffmaps/atc/{args.build_type}",
            save_fig_folder=f"save_fig_{args.build_type}")
    elif args.build_type == "all":
        buildMoD = AllBuildMoD(
            config_file="config_atc.yaml",
            current_cliff=None,
            output_cliff_folder=f"cliffmaps/atc/{args.build_type}",
            save_fig_folder=f"save_fig_{args.build_type}",
            build_type=args.build_type)
    elif args.build_type == "interval":
        buildMoD = IntervalBuildMoD(
            config_file="config_atc.yaml",
            current_cliff=None,
            output_cliff_folder=f"cliffmaps/atc/{args.build_type}",
            save_fig_folder=f"save_fig_{args.build_type}",
            build_type=args.build_type)


    for hour in range(9, 21, 1):
        data_loader = DataLoader('config_atc.yaml', raw_data_file=f'dataset/atc/split_data_random/atc-1024-{hour}_train.csv')
        observed_traj = data_loader.get_observed_traj_all_area_all_time()
        buildMoD.updateMoD(observed_traj, f"atc_1024_{hour}_{hour + 1}")
        