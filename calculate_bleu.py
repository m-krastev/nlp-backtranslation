import argparse

from evaluate import load

def calculate_bleu(hyps: list[str], refs: list[str]):
    sacrebleu = load("sacrebleu")
    bleu = sacrebleu.compute(predictions=hyps, references=refs)
    return bleu


def main():
    
    parser = argparse.ArgumentParser(description="Calculate BLEU score")
    parser.add_argument("hyps", type=str, help="The hypothesis file.")
    parser.add_argument("refs", type=str, help="The reference file.")
    args = parser.parse_args()


    with open(args.hyps, "r") as f:
        hyps = f.readlines()
        
    with open(args.refs, "r") as f:
        refs = f.readlines()
        
    print(calculate_bleu(hyps, refs))
    
if __name__ == "__main__":
    main()
