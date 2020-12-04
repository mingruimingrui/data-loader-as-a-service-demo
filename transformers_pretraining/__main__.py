import argparse
from types import ModuleType
from typing import Dict

from transformers_pretraining.bin import serve_dataset, train_roberta

SUBCOMMANDS: Dict[str, ModuleType] = {
    'serve_dataset': serve_dataset,
    'train_roberta': train_roberta,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--local_rank', type=int, default=-1,
        help='Rank of this process for distributed training.')

    subparsers = parser.add_subparsers(
        dest='subcommand',
        title='subcommands',
        description='Available subcommands',
    )
    for subcommand, module in SUBCOMMANDS.items():
        subparser = subparsers.add_parser(
            subcommand,
            help=module.__doc__,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        module.add_options(subparser)

    args = parser.parse_args()
    module = SUBCOMMANDS[args.subcommand]
    module.main(args)


if __name__ == "__main__":
    main()
