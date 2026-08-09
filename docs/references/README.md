# References

Papers and specifications this pipeline actually rests on, stored so a claim
in the code can be checked against its source without a working network and
without a subscription.

**Only openly available material is stored here.** Several of the works below
are paywalled; those carry a citation and a DOI or URL and nothing else. Do
not add copies of them to this directory.

Cited from the code by file name, so a rename breaks a trail — prefer adding
to this list over reorganising it.

## Stored

| File | What it is | Where it is used |
|---|---|---|
| `2008-Galkin-Reinisch-ARTIST5-INAG.pdf` | Galkin, I. A. and Reinisch, B. W., *The New ARTIST 5 for all Digisondes*, INAG Bulletin, UMLCAR, 2008 | `muf/pick.py` — `DEFAULT_MAX_RANGE_SLOPE`. The "good continuation" grouping principle, and the polarization-tag caveat |
| `2008-Reinisch-SAOXML-5.0-specification.pdf` | Reinisch, B. W., Galkin, I. A. and Khmyrov, G., *SAO.XML 5.0 specification v1.0*, UMLCAR, 2008 | `muf/export/saoxml.py` — the interchange format `muf export` writes |
| `2022-ARTIST-autoscaling-confidence-scores-RSL.pdf` | *ARTIST Ionogram Autoscaling Confidence Scores: Best Practices*, URSI Radio Science Letters, Vol. 4 | Autoscaling confidence, and rejection rates by latitude |
| `2025-IONORT-ISP-WC-raytracing.pdf` | Ray-tracing with IONORT-ISP-WC, arXiv:2506.24098 | `README.md` §LOF — the E-region LOF moving 4.2 → 10.0 MHz once collisions are included |
| `2021-OIIDN-North-China-TIDs.html` | Oblique-Incidence Ionosonde Detection Network, North China — TID observations, PMC7867239 | Oblique network practice; separated transmitter and receiver |
| `DIDBase-rules-of-the-road.html` | Rules of the Road for LGDC / DIDBase data access, UMLCAR | `muf/reference/giro.py` — the terms this pipeline queries DIDBase under |

## Cited but not stored

Paywalled or otherwise not redistributable. Listed so the claim is traceable.

- **Ippolito, A. et al., "Oblique Ionograms Automatic Scaling Algorithm OIASA
  applied to the ionograms recorded by the Ebro observatory ionosonde"**,
  *J. Space Weather Space Clim.* 8, A10 (2018).
  <https://doi.org/10.1051/swsc/2017040> — the parabola-to-the-nose method
  `muf/fit.py` follows. The publisher returns 403 to automated retrieval.
- **Ding, Z. et al., "A method for identification of the F2 layer in noisy
  ionograms for autoscaling of foF2"** — states the continuity criterion as
  "the continuity of the slope of the single layer trace and rejection of
  impractical changes in slope when the ionogram is traversed in the frequency
  axis". Quoted in `muf/pick.py`.
- **Turley, "Ionogram RFI Rejection Using an Autoregressive Interpolation
  Process"**, *Radio Science* (2019).
  <https://doi.org/10.1029/2018RS006683> — interference rejection in
  ionograms. Wiley returns 402.
- **"Ionosphere parameters from verticalized oblique ionograms across Italy"**,
  *Advances in Space Research* (2026).
  <https://doi.org/10.1016/j.asr.2026.01.038> — verticalisation via Martyn's
  equivalent path theorem, the standard route from an oblique ionogram to
  ionospheric parameters.
- **UAG-23A**, URSI Handbook of Ionogram Interpretation and Reduction, 1972 —
  qualifying letters `D`, `E`, `U` and the Standard-vs-Operational MUF
  distinction. Cited throughout `muf/export/saoxml.py`.
- **Blagov, UAG-104**, INAG reference on oblique sounding with signal levels.
  <https://www.ursi.org/files/CommissionWebsites/INAG/uag-104/text/blagov.html>
  — names MOF, MUF and LOF as the three oblique parameters, which is why this
  pipeline reports LOF rather than LUF.
- **ITU-R P.533-13** §9 and eq. (20) — the LUF definition this deliberately
  does not claim to compute, and the absorption term `BACKLOG.md` §15 records
  as unimplemented because the Recommendation publishes its inputs as figures.

## Retrieval

Stored copies were fetched on 2026-08-09. To refresh:

```bash
curl -L -o docs/references/2008-Galkin-Reinisch-ARTIST5-INAG.pdf \
  https://www.ursi.org/files/CommissionWebsites/INAG/web-69/2008/artist5-inag.pdf
```

Sources for the rest are in the tables above.
