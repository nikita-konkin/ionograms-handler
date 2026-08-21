"""Forecasting: run a trained model against the tracked series, store the result.

The models themselves live in a separate project with its own research
lifecycle (``architecture.md`` sec. 7). This package is the thin service above
them: it reads the database, builds the frame a model expects, runs it without
retraining it, and writes ``forecast`` rows.

Three kinds of artifact reach it, and they differ only at import:

``legacy``
    joblib and Keras files produced before this service existed. Their input
    contract is recovered from the artifact rather than declared -- see
    :mod:`services.prediction.artifacts`.
``trained``
    written by the training job, which records the contract as it saves.
``imported``
    dropped on the models volume by an operator and registered by hand.

**Nothing here ever calls ``fit``.** The code this replaces did -- both
``xgb_evaluate`` and ``xgb_test`` in the source project load a saved model and
immediately refit it, so the artifact acts as a hyperparameter carrier and the
returned numbers come from a model trained seconds earlier. That is a
defensible thing for an experiment harness to do and a silent disaster for a
service, because it looks exactly like inference from the outside.
:mod:`services.prediction.infer` is written so the mistake cannot recur, and
``tests/test_prediction_infer.py`` makes ``fit`` raise to prove it.
"""
