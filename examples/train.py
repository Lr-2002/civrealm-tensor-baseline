"""Train an algorithm."""
import argparse
import json
from civtensor.utils.configs_tools import get_defaults_yaml_args, update_args


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo"],
        help="Algorithm name. Choose from: ppo.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="freeciv_tensor_env",
        choices=["freeciv_tensor_env"],
        help="Environment name. Choose from: freeciv_tensor_env.",
    )
    parser.add_argument(
        "--exp_name", type=str, default="installtest", help="Experiment name."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict
    # print(args)
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        args["exp_name"] = all_config["main_args"]["exp_name"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line
    env_args['task_name'] = 'fullgame_50_turn_objectives'
    # print(env_args)

    # start training
    from civtensor.runners.runner import Runner

    runner = Runner(args, algo_args, env_args)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()
