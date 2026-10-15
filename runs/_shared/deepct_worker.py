"""Run inside the separate TensorFlow environment, not the BEIR/PyTorch process."""
import json
import sys
from pathlib import Path


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    import tensorflow as tf
    tf.compat.v1.disable_v2_behavior()
    from deepct import run_deepct
    flags = run_deepct.FLAGS
    flags(["deepct-worker"])
    values = {
        "task_name": "beir", "do_train": False, "do_eval": False, "do_predict": True,
        "data_dir": config["corpus"], "vocab_file": config["vocab"],
        "bert_config_file": config["bert_config"], "init_checkpoint": config["checkpoint"],
        "max_seq_length": config["max_length"], "predict_batch_size": config["batch_size"],
        "output_dir": config["output"], "output_file": "deepct.jsonl", "use_tpu": False,
        "m": config["scaling"], "smoothing": config["smoothing"], "keep_all_terms": config["keep_all_terms"],
    }
    for key, value in values.items():
        setattr(flags, key, value)
    run_deepct.main(None)
    (Path(config["output"]) / "runtime.json").write_text(json.dumps({
        "tensorflow": tf.__version__, "python": sys.version,
        "gpu": [device.name for device in tf.config.list_physical_devices("GPU")],
    }, indent=2))


if __name__ == "__main__":
    main()
