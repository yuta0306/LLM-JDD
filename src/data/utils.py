from pathlib import Path
from typing import Literal
import json
from sklearn.model_selection import train_test_split

def load_data(
    input_path: str,
    task: Literal["JDD"] = "JDD",
    test_size: int = 0.1,
    seed: int = 42
) -> tuple[list[dict]]:
    top = Path(input_path)
    
    train_data = []
    eval_data = []
    if task == "JDD":
        for path in sorted(top.glob("*.json")):
            with open(path, "r") as f:
                data = json.load(f)
            train_data_topic, eval_data_topic = train_test_split(
                data,
                test_size=test_size,
                random_state=seed,
                shuffle=True,
            )
            train_data.extend(train_data_topic)
            eval_data.extend(eval_data_topic)
                
    return train_data, eval_data
