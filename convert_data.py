"""
Convert data from our format to the one used by Li et al.

"""
import argparse
import json
from tqdm import tqdm
import os


def run(args):
    data = []
    for line in tqdm(open(args.in_path), "Processing datum"):
        datum = json.loads(line)
        new_datum = transform(datum)
        data.append(new_datum)

    out_dir = os.path.dirname(args.out_path)
    if not os.path.exists(out_dir):
        print("Creating dir", out_dir)
        os.makedirs(out_dir)
    with open(args.out_path, "w") as f:
        json.dump(data, f)

    print("All done")


def transform(datum):
    """ The formats are super similar. Main diff is there ends are inclusive"""
    new_datum = {
        "sentence": datum["tokens"],
        "labeled entities": [(d["start"], d["end"] - 1, d["type"]) for d in datum["gold_annotations"]],
    }
    return new_datum


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-path", type=str)
    parser.add_argument("--out-path", type=str)
    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args()
    run(args)