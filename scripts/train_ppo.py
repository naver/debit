from habitat_baselines.run import build_parser, run_exp
import debit


if __name__ == "__main__":
    parser = build_parser()
    parser.set_defaults(run_type="train", exp_config="configs/imgnav-gibson-debit.yaml")
    args = parser.parse_args()
    run_exp(**vars(args))
