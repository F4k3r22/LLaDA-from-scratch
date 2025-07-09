from configs_llada import LLaDAConfig
from model import LLaDAModel

# We will train and iterate from the base with a 100M parameter model to test and then scale to the 1B model.
config = LLaDAConfig.from_json_file("config.json")
model = LLaDAModel(config, init_params=True)