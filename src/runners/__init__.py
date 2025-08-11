REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .evaluation_runner import EvalRunner
REGISTRY["eval"] = EvalRunner

from .collection_runner import CollectRunner
REGISTRY["col"] = CollectRunner

from .advcol_runner import AdversarialCollectorRunner
REGISTRY["advcol"] = AdversarialCollectorRunner

from .adversarial_runner import AdversarialRunner
REGISTRY["adv"] = AdversarialRunner

from .col_parallel_runner import ParallelCollector
REGISTRY["parlCol"] = ParallelCollector