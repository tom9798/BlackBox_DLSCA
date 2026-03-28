"""Attack script dedicated to the general_masking model in train_models_general.py.

This script mirrors the attack flow from attack_conf.py but keeps only the
general_masking path to avoid ambiguity with older models.
"""

import argparse
import numpy as np
from tqdm import tqdm

from utility import (
    read_from_h5_file,
    get_hot_encode,
    load_model_from_name,
    get_rank,
    get_pow_rank,
    get_rank_list_from_prob_dist,
    XorLayer,
    InvSboxLayer,
    METRICS_FOLDER,
)
from train_models_general import general_masking


class Attack:
    def __init__(
        self,
        model_name: str,
        n_experiments: int = 1000,
        n_traces: int = 10000,
        resnet: bool = False,
    ):
        """Prepare attack for the general_masking model.

        Args:
            model_name: Basename used at training time (weights file is `<model_name>_all.weights.h5`).
            n_experiments: Number of attack experiments.
            n_traces: Number of traces to load from the attack set.
            resnet: Whether the attacked model uses the resnet backbone.
        """

        self.model_name = model_name
        self.n_experiments = n_experiments
        self.traces_per_exp = 100

        # Load attack traces and metadata
        traces, labels_dict, metadata = read_from_h5_file(
            n_traces=n_traces, dataset="attack", load_plaintexts=True
        )
        self.traces = traces
        self.n_traces = traces.shape[0]
        input_length = traces.shape[1]

        # Build architecture and load weights
        model = general_masking(input_length=input_length, resnet=resnet)
        weights_name = f"{self.model_name}_all.weights.h5"
        model = load_model_from_name(model, weights_name)

        # Prepare predictions
        self.powervalues = np.expand_dims(traces, 2)
        predictions = model.predict({"traces": self.powervalues})

        # Plaintexts one-hot for xor unmasking
        plaintexts = np.array(metadata["plaintexts"], dtype=np.uint8)[: self.n_traces]
        self.plaintexts = get_hot_encode(plaintexts)

        # Convert predictions to key-probabilities following attack_conf logic
        xor_op = XorLayer(name="xor")
        inv_op = InvSboxLayer(name="iinv")
        self.predictions = np.empty((self.n_traces, 14, 256), dtype=np.float32)

        batch_size = 1000
        for batch in range(0, self.n_traces // batch_size):
            print(f"Batch of prediction {batch + 1} / {self.n_traces // batch_size}")
            for byte in tqdm(range(2, 16)):
                # Outputs are masked target (t); xor with plaintext to derive key prob dist
                self.predictions[
                    batch * batch_size : (batch + 1) * batch_size, byte - 2
                ] = xor_op(
                    [
                        predictions[f"output_{byte}"][
                            batch * batch_size : (batch + 1) * batch_size
                        ],
                        self.plaintexts[
                            batch * batch_size : (batch + 1) * batch_size, byte
                        ],
                    ]
                )

        # Fixed master key from attack_conf
        master_key = [
            0x00,
            0x11,
            0x22,
            0x33,
            0x44,
            0x55,
            0x66,
            0x77,
            0x88,
            0x99,
            0xAA,
            0xBB,
            0xCC,
            0xDD,
            0xEE,
            0xFF,
        ]
        self.subkeys = master_key

        # Per-byte accuracy (GeneralArch style)
        print(f"\n=== Per-byte s1 accuracy ===")
        for byte in range(14):
            _, acc, _, _ = get_rank_list_from_prob_dist(
                self.predictions[:, byte],
                np.repeat(self.subkeys[byte + 2], self.predictions.shape[0]),
            )
            print(f"  Byte {byte + 2:2d}: {acc:.2f}%")

        self.history_score = {}

    def run(self):
        from gmpy2 import mpz, mul

        for experiment in range(self.n_experiments):
            self.history_score[experiment] = {"total_rank": []}
            self.subkeys_guess = {i: np.zeros(256,) for i in range(2, 16)}
            for i in range(2, 16):
                self.history_score[experiment][i] = []

            traces_order = np.random.permutation(self.n_traces)[: self.traces_per_exp]
            count_trace = 1

            for trace in traces_order:
                all_recovered = True
                ranks = {}
                total_rank = mpz(1)

                for byte in range(2, 16):
                    self.subkeys_guess[byte] += np.log(
                        self.predictions[trace][byte - 2] + 1e-36
                    )
                    ranks[byte - 2] = get_rank(self.subkeys_guess[byte], self.subkeys[byte])
                    self.history_score[experiment][byte].append(ranks[byte - 2])
                    total_rank = mul(total_rank, mpz(ranks[byte - 2]))
                    if np.argmax(self.subkeys_guess[byte]) != self.subkeys[byte]:
                        all_recovered = False

                self.history_score[experiment]["total_rank"].append(get_pow_rank(total_rank))

                if all_recovered:
                    for elem in range(count_trace, self.traces_per_exp):
                        for i in range(2, 16):
                            self.history_score[experiment][i].append(ranks[i - 2])
                        self.history_score[experiment]["total_rank"].append(0)
                    break

                count_trace += 1

            if experiment % max(1, self.n_experiments // 10) == 0:
                print(f"  Exp {experiment}: final GE = 2^{self.history_score[experiment]['total_rank'][-1]}")

        array_total_rank = np.empty((self.n_experiments, self.traces_per_exp))
        for i in range(self.n_experiments):
            for j in range(self.traces_per_exp):
                array_total_rank[i][j] = self.history_score[i]["total_rank"][j]
        mean_ge = np.mean(array_total_rank, axis=0)
        whe = np.where(mean_ge < 2)[0]
        threshold = int(np.min(whe)) if whe.shape[0] >= 1 else self.traces_per_exp

        print(f"\n=== Fixed-key results ({self.n_experiments} experiments) ===")
        print(f"GE < 2 at trace: {threshold}")
        print(f"Final mean GE: 2^{mean_ge[-1]:.1f}")

        import pickle
        result_path = METRICS_FOLDER + f"history_attack_experiments_{self.model_name}_{self.n_experiments}"
        with open(result_path, "wb") as file:
            pickle.dump(self.history_score, file)
        print(f"\nResults saved to {result_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run attack for general_masking model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_name", type=str, required=True, help="Model basename (e.g., DualMask)")
    parser.add_argument("--n_experiments", type=int, default=10, help="Number of attack experiments")
    parser.add_argument("--n_traces", type=int, default=10000, help="Number of traces to use")
    parser.add_argument("--resnet", action="store_true", default=False, help="Use ResNet variant")
    args = parser.parse_args()

    attack = Attack(
        model_name=args.model_name,
        n_experiments=args.n_experiments,
        n_traces=args.n_traces,
        resnet=args.resnet,
    )
    attack.run()


if __name__ == "__main__":
    main()