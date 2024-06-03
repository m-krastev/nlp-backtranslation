import argparse


def calculate_bleu(hyps: list[str], refs: list[str]):
    from evaluate import load

    sacrebleu = load("sacrebleu")
    bleu = sacrebleu.compute(predictions=hyps, references=refs)
    return bleu


def calculate_comet(hyps: list[str], refs: list[str], src: list[str]):
    from evaluate import load
    from statistics import stdev

    comet = load("comet")
    score = comet.compute(predictions=hyps, references=refs, sources=src)
    score = (
        f"COMET score: {score['mean_score']*100:.2f}±{stdev(score['scores'])*100:.1f}"
    )
    return score


def main():
    parser = argparse.ArgumentParser(description="Calculate BLEU score")
    parser.add_argument("hyps", type=str, help="The hypothesis file.")
    parser.add_argument("refs", type=str, help="The reference file.")
    parser.add_argument("--src", type=str, help="Path to the source data")
    parser.add_argument("--comet", action="store_true")
    args = parser.parse_args()

    with open(args.hyps, "r") as f:
        hyps = f.readlines()

    with open(args.refs, "r") as f:
        refs = f.readlines()

    if args.comet:
        with open(args.src) as f:
            src = f.readlines()
        print(calculate_comet(hyps, refs, src))
    else:
        print(calculate_bleu(hyps, refs))


if __name__ == "__main__":
    main()
