"""Regression: a Dreams-mode e2e store needs practice scenes to train on.

Under learning="dreams" preview() counts train = practice_steps(store),
which comes from the practice table. The harness seeded only recorded
exchanges, so train was 0 and the log stage always failed for vLLM with
"Synthetic conversations did not yield enough independent hindsight".
"""
from scripts.e2e import seed, seed_practice
from songe.config import Config
from songe.log import Store
from songe.sleep.worker import preview


def test_log_stage_has_trainable_pool_under_dreams(tmp_path):
    config = Config(
        learning="dreams", min_train=16, min_heldout=8, min_new=8
    )
    store = Store(tmp_path)
    seed(store, config.public_model)
    seed_practice(store)
    info, _, _ = preview(config, store)
    assert info["train"] >= config.min_train
    assert info["heldout_conversations"] >= config.min_heldout
