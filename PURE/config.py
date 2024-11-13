import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description="PURE_ICML_2024")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased")
    parser.add_argument("--victim_model", type=str, default="bert-base-uncased")

    parser.add_argument("--trigger_planting_dataset", type=str, default="IMDB", help="IMDB, YELP, SST-2")
    parser.add_argument("--clean_dataset", type=str, default="SST-2")

    parser.add_argument("--poisoning_epoch", type=int, default=5)
    parser.add_argument("--defending_epoch", type=int, default=3)
    parser.add_argument("--acc_threshold", type=float, default=0.85)
    parser.add_argument("--prune_step", type=int, default=10)
    parser.add_argument("--penalty_coefficient", type=float, default=0.15)

    parser.add_argument("--planting_mode", type=str, default="DS", help="DS - Domain Shift, FDK - Full Data Knowledge")
    parser.add_argument("--attack_mode", type=str, default="BadNet", help="BadNet, LayerWise, HiddenKiller, StyleBkd")

    parser.add_argument("--max_len_long", type=int, default=256)
    parser.add_argument("--max_len_short", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)

    return parser



