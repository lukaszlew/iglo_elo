# [IGLO](https://iglo.szalenisamuraje.org/) ELO ranking

## Ratings

Model output is in [iglo_elo_table.txt](https://raw.githubusercontent.com/lukaszlew/iglo_elo/main/iglo_elo_table.txt)

100 point difference is 1:2 win odds  (33% vs 66%)  
200 point difference is 1:5 win odds  (20% vs 80%)  
300 point difference is 1:10 win odds  (10% vs 90%)  

## What it is good for?

To see how well did you do in IGLO. Taking into account how well did your opponents did.
This is useful especailly in lower groups that have a substantial overlap in their rating difference.
It is useful to find opponents of with similar track record in IGLO.

## What it is bad for?

Player motivation. 
This system does not have any gamification incentives. It uses data efficiently and converges rather quickly.
It can be compared to hidden MMR used for match-making in games like Starcraft 2, not to the player-visible motivating points with many bonuses.

## Model details

The model finds a single number (ELO strength) for each player.
Given ELO of two players, the model predicts probability of win in a game:
If we assume that $P$ and $Q$ are rankings of two players, the model assumes:

$$P(\text{P vs Q win}) = \frac{2^P}{2^P + 2^Q} = \frac{1}{1 + 2^{Q-P}} $$

This means that if both rankings are equal to $a$, then: $P(win) = \frac{2^a}{2^a+2^a} = 0.5$.
If a ranking difference is one point, we have $P(win) = \frac{2^{a+1}}{2^{a+1}+2^{a}} = \frac{2}{2+1}$
Two point adventage yields $P(win) = \frac{1}{5}$
$n$ point adventage yields $P(win) = \frac{1}{1+2^n}$

For readability reasons we rescale the points by 100. This is exactly equivalent to using this equation:

$$ \frac{1}{1 + 2^{\frac{Q-P}{100}}} $$


## Comparison to chess ELO

Chess ELO is using $10^0.25 \approx 1.77$ instead of 2 as the basis of the exponent. Effectively using this equation for model:
$$ \frac{1}{1 + 10^{\frac{Q-P}{400}}} $$
But I thought round numbers $1/3, 1/5, 1/10$ will be easier to remember.

Also chess ELO uses data from a game only once, while WHR (this implementation) iterates over the data many times allowing 
for much more efficient data use. 

## Comparison to EGD

EGD is trying to match 100 points to one 'dan/kyu' rank, instead of fixed probability of win (such as 1/3 for 100 points here).
And probability gap is [higher at higher level](https://www.europeangodatabase.eu/EGD/winning_stats.php), e.g.:

- $$P(\text{3dan vs 5dan win}) \approx 16%$$ (1:5 odds)
- $$P(\text{15kyu vs 13kyu win}) \approx 33%$$ (1:2 odds)

EGD implements this by varying the exponent base for different ratings.

## What's implemented

This repo implements:

- Harvesting game results data from IGLO API.
- Implement [Bradley-Terry (BT) model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) for player ranking (better variant of ELO).
- Generalization BT model to a variant of [WHR model](https://www.remi-coulom.fr/WHR/) with time as seasons.
- Some simple plots.
- JSON export (look for comments in the code).


## ToDo

- distribution fit to account for heavy tail
- maybe use the same units as Chess?
- maybe use variable units as EGD?

### Ideas

- Fit player variance - not clear what equations.
- Fit rank confidence - not clear what equations.
