# IGLO ELO ranking

This repo implements:

- Harvesting game results data from IGLO API.
- Implement [Bradley-Terry (BT) model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) for player ranking (better variant of ELO).
- Generalization BT model to a variant of [WHR model](https://www.remi-coulom.fr/WHR/) with time as seasons.
- Some simple plots.
- JSON export (look for comments in the code).

## Ratings

Model output is in [iglo_elo_table.txt](https://raw.githubusercontent.com/lukaszlew/iglo_elo/main/iglo_elo_table.txt)

100 point difference is 1:2 win odds  (33% vs 66%)
200 point difference is 1:5 win odds  (20% vs 80%)
300 point difference is 1:10 win odds  (10% vs 90%)

## Model details

The model finds a single number (ELO strength) for each player.
Given ELO of two players, the model predicts probability of win in a game:
If we assume that $P$ and $Q$ are rankings of two players, the model assumes:

$$P(P wins vs Q) = \frac{2^P}{2^P + 2^Q}$$

This means that if both rankings are equal to $a$, then: $P(win) = \frac{2^a}{2^a+2^a} = 0.5$.
If a ranking difference is one point, we have $P(win) = \frac{2^{a+1}}{2^{a+1}+2^{a}} = \frac{2}{2+1}$
Two point adventage yields $P(win) = \frac{1}{5}$
$n$ point adventage yields $P(win) = \frac{1}{1+2^n}$

Chess ELO is using $10^0.25 \approx 1.77$ instead of 2 as the basis of the exponent, but I thought round numbers $1/3, 1/5, 1/9$ will be easier to remember.

Both chess ELO and Iglo ELO rescales 1 point to 100.

## ToDo

- distribution fit to account for heavy tail
- accounting for player growth - linear growth coefficient for players

### Ideas

- Fit player variance - not clear what equations.
- Fit rank confidence - not clear what equations.
