# [IGLO](https://iglo.szalenisamuraje.org/) ELO ranking

This implementation of a very multi-pass accurate rating model for IGLO.

## Ratings

Model output is in [iglo_elo_table.txt](https://raw.githubusercontent.com/lukaszlew/iglo_elo/main/iglo_elo_table.txt)

100 point difference is 1:2 win odds  (33% vs 66%)  
200 point difference is 1:5 win odds  (20% vs 80%)  
300 point difference is 1:10 win odds  (10% vs 90%)  

This is different that EGD, where 100 points of rating is not fixed winning ods!

## What it is good for?

To see how well did you do in IGLO. Taking into account how well did your opponents did.
This is useful especailly in lower groups that have a substantial overlap in their rating difference.
It is useful to find opponents of with similar track record in IGLO.

## What it is bad for?

Player motivation. 
This system does not have any gamification incentives. It uses data efficiently and converges rather quickly.
It can be compared to hidden MMR used for match-making in games like Starcraft 2, not to the player-visible motivating points with many bonuses.

## If it so accurate why other systems are not using it?

Typical ELO systems (like EGD) use every game result only once and update rating based on it.
This model did 1500 passes over the data until it converged (the first pass returns numbers more or less like standard ELO system, equations are almost the same).
However so many passes is too expensive when the number of games is as big as EGD.

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

## Comparison to single-pass systems.

In single pass systems, if you play a game, it will not affect the model estimation of your rating yesterday.
In multi-pass system we can estimate every player's rating for every season (or even every day).
Then iteratively pass over the data again and again until we find rating curves that best fit the data.

There is a parameter that controlls how fast the rating can change. 
WHR model assumes that player rating is a gaussian process in time, and this parameter is variance of this gaussian process.

The consequence is that data flows both time directions: if you play game in season 20, it will also affect your ratings for season 19 (a bit) and season 18 (very small bit) etc.

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

## Can I convert these ratings to EGD?

Because EGD is using different exponent base, it is not that easy to convert directly.
These must be a monotonic conversion function but it is non-linear, and I don't know how to derive the formula.

The best you can do is compare win probabilities (this is universal unit as opposed to dan/kyu system).

The code could be updated to use the same dynamic exponent base as EGD and re-run to converge again to ratings that are comparable to EGD, but it takes some amount of programming and experimenting.

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
