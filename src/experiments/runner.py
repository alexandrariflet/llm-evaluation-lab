from itertools import product
import pandas as pd


class ExperimentRunner:
    def __init__(self, llm, embedder, build_prompt_fn):
        self.llm = llm
        self.embedder = embedder
        self.build_prompt = build_prompt_fn

    # ----------------------------
    # Config builder
    # ----------------------------
    def build_config(self, temperature, top_p, top_k, repeat_penalty):
        return {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "num_predict": 200
        }

    # ----------------------------
    # Single run (n samples per condition)
    # ----------------------------
    def run_single(self, prompt, config, seed, n=5):
        outputs = []

        for i in range(n):
            out = self.llm.generate(
                prompt,
                seed=seed + i,
                **config
            )
            outputs.append(out)

        return outputs

    # ----------------------------
    # Full grid search
    # ----------------------------
    def run_grid(
        self,
        languages,
        temperatures,
        top_ps,
        top_ks,
        repeat_penalties,
        seeds,
        n=5
    ):
        results = []

        for lang, temp, top_p, top_k, rp, seed in product(
            languages,
            temperatures,
            top_ps,
            top_ks,
            repeat_penalties,
            seeds
        ):
            prompt = self.build_prompt(lang)

            config = self.build_config(
                temperature=temp,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=rp
            )

            outputs = self.run_single(
                prompt=prompt,
                config=config,
                seed=seed,
                n=n
            )

            results.append({
                "language": lang,
                "temperature": temp,
                "top_p": top_p,
                "top_k": top_k,
                "repeat_penalty": rp,
                "seed": seed,
                "outputs": outputs
            })

        return results

    # ----------------------------
    # Flatten for analysis
    # ----------------------------
    def to_dataframe(self, results):
        rows = []

        for r in results:
            for text in r["outputs"]:
                rows.append({
                    "language": r["language"],
                    "temperature": r["temperature"],
                    "top_p": r["top_p"],
                    "top_k": r["top_k"],
                    "repeat_penalty": r["repeat_penalty"],
                    "seed": r["seed"],
                    "text": text
                })

        return pd.DataFrame(rows)