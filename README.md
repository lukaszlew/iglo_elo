# IGLO ELO ranking

This repo implements:

- Harvesting game results data from IGLO API.
- Implement [Bradley-Terry (BT) model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) for player ranking (better variant of ELO).
- Generalization BT model to a variant of [WHR model](https://www.remi-coulom.fr/WHR/) with time as seasons.
- Some simple plots.
- JSON export (look for comments in the code).

## ToDo

- dots in the plot
- distribution fit to account for heavy tail
- accounting for player growth - measure downdrift and make it flat.
- accounting for player growth - linear growth coefficient for players

### Ideas

- Fit player variance - not clear what equations.
- Fit rank confidence - not clear what equations.
