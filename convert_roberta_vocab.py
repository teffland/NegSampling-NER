""" Convert roberta vocab from dict to line separated flat file so their vocab loader works.
"""
import json


def run():
    vocab = json.load(open("resource/roberta-base/vocab.json"))
    ordered = [t for t, _ in sorted(vocab.items(), key=lambda x: x[1])]
    with open("resource/roberta-base/vocab.txt", "w") as f:
        f.write("\n".join(ordered))


if __name__ == "__main__":
    run()