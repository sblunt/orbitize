Gaia-4 radial velocities: Stefansson et al. (2025)
The Astronomical Journal, 169, 107
DOI: https://doi.org/10.3847/1538-3881/ada9e1
CDS catalogue: J/AJ/169/107
Source table: https://cdsarc.cds.unistra.fr/ftp/J/AJ/169/107/table5.dat
Original documentation: https://cdsarc.cds.unistra.fr/ftp/J/AJ/169/107/ReadMe

gaia4_rv_stefansson2025_table5.dat preserves the original published table.
Its columns are BJD(TDB) [days], RV [m/s], RV uncertainty [m/s], instrument.

gaia4_rv_stefansson2025.csv is sorted by epoch and readable by
orbitize.read_input.read_file(). It contains 23 primary-star RVs:
10 HPF, 8 NEID and 5 FIES.

Conversion:
- epoch = astropy.time.Time(BJD, format="jd", scale="tdb").tcb.mjd.
  MJD(TCB) matches the Gaia EAS epoch time scale. Conversion changes
  these timestamps by approximately 22.6-23.3 seconds relative to
  simply subtracting 2400000.5 from BJD(TDB).
- rv and rv_err are divided by 1000 to give km/s.
- object=0 denotes the primary star.
- bjd_tdb preserves the original published timestamp.
- Instrument identifiers and formal uncertainties are preserved.
- No median-centering, offset subtraction, resampling or added jitter.

Fit separate RV zero points and any additional jitter for each instrument.
The data are measurements, not a published posterior or an orbital prior.
Retrieved 2026-10-10. Existing notebooks and library files were not edited.
