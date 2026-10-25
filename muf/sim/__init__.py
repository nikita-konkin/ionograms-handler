"""An HF channel simulator, tuned from chirp soundings.

The idea, from the dissertation this grew out of, is a hybrid channel model
whose parameters come from sounding: modes after Watterson, Doppler, the
dispersion within each mode, the magneto-ionic split -- and a state that
holds them per mode as curves over frequency, so that the same state can be
sounded and can carry a signal.

* :mod:`.state`     the channel state, with every value's provenance;
* :mod:`.observe`   a state from a sounding, through muf's own trace;
* :mod:`.presets`   ITU-R F.1487's test channels, and a mode from a law;
* :mod:`.fading`    Watterson tap gains;
* :mod:`.channel`   a signal through the state at one carrier;
* :mod:`.sounding`  a synthetic ``.lfs`` of the state;
* :mod:`.check`     the closed loop: does a state survive being sounded?

``muf sim`` drives all of it; docs/simulator.md describes it.
"""

from .state import ChannelState, Circuit, Mode, Provenance, Sweep

__all__ = ["ChannelState", "Circuit", "Mode", "Provenance", "Sweep"]
