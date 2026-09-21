---
title: Which official datasets ORCA uses and what each one is good for
agency: ORCA data inventory
tier: seed
tags: catalog, datasets, mosdac, incois, imd, discovery
url: https://mosdac.gov.in/
---

# ISRO / MOSDAC

MOSDAC, run by the Space Applications Centre, publishes a THREDDS data server at
`mosdac.gov.in/live_data` that needs no login. ORCA uses it for:

- **Ocean State Forecast, circulation (`OSF_CIRC`).** Daily files at 10 km,
  6-hourly out to 5 days. Sea temperature, salinity, mixed layer depth and
  surface currents. This is ORCA's authoritative sea surface temperature and
  current source.
- **Ocean State Forecast, waves (`OSF_WAVE`).** Significant wave height, mean
  wave period and direction, primary and secondary swell, and model winds.
- **PFZ input grids (`pfz/sst`, `pfz/chl`).** The sea surface temperature and
  chlorophyll grids behind the PFZ advisory service.
- **GSMaP ISRO rainfall.** Hourly satellite rainfall.
- **INSAT-3D, 3DR and 3S products.** Imager and sounder products including sea
  surface temperature, outgoing longwave radiation, cloud mask and cyclone
  imagery.
- **EOS-06 (Oceansat-3) scatterometer analysed winds.** Ocean surface wind
  vectors.

# INCOIS / Ministry of Earth Sciences

- **INCOIS ERDDAP** at `erddap.incois.gov.in` is open and machine readable.
  It carries Oceansat-2 OCM ocean colour, IRS-P4 OCM chlorophyll, weekly Argo
  sea surface temperature, 10-day and monthly Argo temperature and salinity
  analyses, daily and monthly ASCAT winds, and the Indian Argo float table.
- **Potential Fishing Zone advisory** and **Ocean State Forecast** bulletins are
  published as web pages and PDFs for each maritime state.
- INCOIS also runs the tsunami early warning centre for the Indian Ocean and
  issues high wave alerts for the coast.

# IMD

IMD is the authority for weather warnings, and RSMC New Delhi is the authority
for tropical cyclones in the north Indian Ocean. Its sub-division wise warning
page carries the operative wording for coastal warnings, including the standard
"fishermen are advised not to venture into the sea" advisory. IMD's structured
JSON API requires a key issued by the department.

# What is not available as open data

- Lightning strike locations. IMD's Damini network is distributed through a
  mobile app, not an API. Any lightning risk figure ORCA gives is a proxy
  computed from convective instability and rainfall, and is labelled as such.
- The daily PFZ advisory as structured data. It is published as per-state
  bulletins for human readers.
- Gazetted protected-area and maritime boundary geometry as an open service.
  INCOIS hosts these layers but its GeoServer blocks non-browser clients.

# Choosing a source

Order of preference for any variable: an ISRO product on MOSDAC, then an INCOIS
product, then IMD for warnings, and only then a non-Indian model. ORCA labels
every value with which tier it came from, so a user can see when an answer rests
on a fallback rather than on an official Indian product.
