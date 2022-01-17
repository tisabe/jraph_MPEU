import ml_collections
from typing import Dict, Iterable
import jraph

def get_datasets(config: ml_collections.ConfigDict
) -> Dict[str, Iterable[jraph.GraphsTuple]]