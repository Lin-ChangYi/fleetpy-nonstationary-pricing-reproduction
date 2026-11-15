# Data provenance and preprocessing

## Historical source

The public FleetPy Manhattan package is the direct simulation-data source. Its historical request data are derived from the public NYC Taxi and Limousine Commission (TLC) trip-record data for 2018-11-11 through 2018-11-18. FleetPy filters the records to trips with both endpoints in Manhattan and matches the endpoints to the Manhattan road network. This experiment uses the full day `2018-11-12.csv` from that package, containing 213,996 matched requests before study-specific sampling.

The source package also supplies the OpenStreetMap-based Manhattan network, preprocessed free-flow travel-time and distance matrices, an hourly mean traffic factor for 2018-11-12, the Manhattan Taxi-Zone node mapping, and vehicle definitions. No confidential trip-level data are used or redistributed.

## Request preparation

`tools/prepare_paired_inputs.py` applies the following deterministic procedure.

1. It augments every `2018-11-12.csv` request with `max_fare`. For request identifier `i`, the value is a log-normal draw with median 3,600 simulation fare units, log-scale standard deviation 0.5, lower bound 100, and a deterministic SplitMix64/Box--Muller transformation using salts 42 and 43. The same request ID always receives the same value.
2. It reconstructs 50 frozen daily traces of exactly 10,000 unique requests. The 48 half-hour request quotas for every paired sample are recorded in `config/paired_samples_50.csv`; therefore the historical intraday request-arrival profile is preserved in each trace.
3. Within every half-hour period, 70% of the selected origins are drawn from the union of six fixed grid-boundary corridors: vertical coordinates 585,000, 586,500, and 588,000 m and horizontal coordinates 4,510,500, 4,512,000, and 4,513,500 m in the network coordinate system. Each corridor has half-width 375 m, using the L-infinity distance convention. The remaining requests are selected from the non-corridor population; no request is duplicated.
4. It builds a 900-second Taxi-Zone perfect-trip forecast by aggregating selected origin and destination trips after adding the FleetPy travel time from the precomputed matrix.

The manifest records SHA-256 hashes for each generated demand and forecast file. Passing `--verify-hashes` checks the reconstructed files against the inputs used for the completed 50-pair evaluation.

## Nonstationary passenger-response environment

The request-arrival trace is fixed before simulation. Nonstationarity in price response is introduced by the five pre-specified binary-logit phases in `config/nonstationary_p3_r2.yaml`:

| Time | Intercept | Fare sensitivity |
| --- | ---: | ---: |
| 00:00-07:00 | 1.60 | 10.00 |
| 07:00-10:00 | 2.20 | 0.75 |
| 10:00-16:00 | 3.30 | 9.10 |
| 16:00-20:00 | 1.60 | 10.00 |
| 20:00-24:00 | 2.20 | 0.75 |

The acceptance model uses the quoted FleetPy fare divided by the request-specific `max_fare`, the quoted pickup wait in minutes, a wait coefficient of 0.10, a 3-minute reference wait, and probability bounds of 0.02 and 0.98. These coefficients are fixed before simulation; no learning method estimates them.

## Paired simulation design

Every method in a paired sample receives the same frozen request trace, forecast, FleetPy random seed, and MAB random seed. The simulation has 400 single-occupancy vehicles, runs for 24 hours, updates the pricing assignment every 1,800 seconds, and selects among global fare multipliers 0.8, 1.0, and 1.2. The pricing grid contains 42 square 1,500 m cells. The controller resolves the interference radius to one quarter of the grid edge, 375 m, when no override is specified.

Realized operator revenue is the analysis endpoint. A fare contributes only after a rider accepts the offer and the ride is completed; rejected or unserved requests contribute zero.
