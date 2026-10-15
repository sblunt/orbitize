# Gaia DR4 prerelease epoch astrometry (EAS, June 2026)

Data credit: **ESA/Gaia/DPAC**.

`GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml` is an unchanged copy from
`gaia-dr4-prerelease-epoch-astrometry_2026-06-26`. It contains real prerelease
observations of 12 sources in a native VOTable, not simulated injection data.
The file has 1,008 transit rows with per-CCD measurement arrays; it has not been
filtered or averaged for distribution here.

The [HD 114762 tutorial](../../docs/tutorials/Gaia_DR4_EAS.ipynb) selects source
`3937211745905473024` with `read_input.from_dl2` and retains 558 AGIS-used CCD
measurements across 63 transits. The notebook lives in `docs/tutorials`; this
directory holds its input data.

`hd_114762_rv_kane2011.csv` contains the 74 Lick/Hamilton radial velocities
published in Table 1 of Kane et al. (2011). The table dates were converted from
JD - 2440000 to MJD, and the velocities and uncertainties were converted from
m/s to km/s. The five detector dewars are retained as separate instruments so
the orbit fit can assign each one its own zero point and jitter. The numerical
measurements are otherwise unchanged.

## Provenance and terms

These are the source, license, and credit links supplied with the original download:

- [ESA DR4 prerelease information](https://www.cosmos.esa.int/web/gaia/dr4-prerelease)
- [Gaia data license](https://www.cosmos.esa.int/web/gaia-users/license)
- [Official acknowledgement and citation instructions](https://gea.esac.esa.int/archive/documentation/GDR3/Miscellaneous/sec_credit_and_citation_instructions/)

The observations were collected by [ESA's Gaia mission](https://www.cosmos.esa.int/gaia)
and processed by [DPAC](https://www.cosmos.esa.int/web/gaia/dpac/consortium).
DPAC's support comes from national institutions, especially those participating
in the Gaia Multilateral Agreement. The data retain ESA's terms; bundling them
with orbitize does not relicense them under the software license.

For publications, follow the linked official acknowledgement wording and citation
guidance. The download points to the DR3 credit page, but this XML is a **DR4
prerelease**, not a DR3 epoch-astrometry product.

## References

- Gaia Collaboration, Prusti et al. (2016), *The Gaia mission*, A&A 595, A1.
  [doi:10.1051/0004-6361/201629272](https://doi.org/10.1051/0004-6361/201629272).
- Gaia Collaboration, Vallenari et al. (2023), *Gaia Data Release 3: Summary of
  the content and survey properties*, A&A 674, A1.
  [doi:10.1051/0004-6361/202243940](https://doi.org/10.1051/0004-6361/202243940).
  This is the DR3 release reference named by the supplied credit instructions,
  not a publication describing this DR4 prerelease.
- Winn (2022), the HD 114762 orbital comparison used in the tutorial:
  [arXiv:2209.05516](https://arxiv.org/abs/2209.05516).
  These published parameters are a comparison, not injection truth.
- Kane et al. (2011), *Revised Orbit and Transit Exclusion for HD 114762b*,
  ApJ 735, L41. [doi:10.1088/2041-8205/735/2/L41](https://doi.org/10.1088/2041-8205/735/2/L41),
  [arXiv:1106.1434](https://arxiv.org/abs/1106.1434).
