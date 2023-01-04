# ToDo:
# - per season gaussian process
# - plot
# - heavy tail fit added
# - Plan how to present
#   - present partial first.
#   - Short writup in markdown
#   - plot
# Later:
# - player variance - not clear how it works
# - rank confidence - not clear what equations

import jax.numpy as jnp
import jax
from jax.tree_util import tree_map
import json
import numpy as np
import matplotlib.pyplot as plt

from jax.config import config
config.update("jax_numpy_rank_promotion", "raise")  # bug prevention
config.update("jax_enable_x64", True)  # better model accuracy


def win_prob(elo, opp_elo):
  return 1.0 / (1.0 + jnp.exp2(opp_elo-elo))
  # This is more understandable and equivalent:
  # return jnp.exp2(elo) / (jnp.exp2(elo) + jnp.exp2(opp_elo))


def log_win_prob(elo, opp_elo):
  # return jnp.log2(win_prob(elo, opp_elo))
  diff = opp_elo - elo
  return -jnp.log2(1.0 + jnp.exp2(diff))


def train(
  data,
  steps,
  do_log=False,
  learning_rate=10000,
  elo_season_stability=0.5,
):
  p1_win_probs = data['p1_win_probs']
  p2_win_probs = 1.0 - p1_win_probs
  p1s = data['p1s']
  p2s = data['p2s']
  seasons = data['seasons']

  player_count = jnp.maximum(jnp.max(p1s), jnp.max(p2s)) + 1
  season_count = jnp.max(seasons) + 1

  (data_size,) = p1s.shape
  assert seasons.shape == (data_size,)
  assert p1s.shape == (data_size,)
  assert p2s.shape == (data_size,)
  assert p1_win_probs.shape == (data_size,)

  def model(params):
    delos = params['elos']
    assert delos.shape == (player_count, season_count)
    p1_elos = delos[p1s, seasons]
    p2_elos = delos[p2s, seasons]

    assert p1_elos.shape == (data_size,)
    assert p2_elos.shape == (data_size,)

    winner_win_prob_log = p1_win_probs * log_win_prob(p1_elos, p2_elos) + p2_win_probs * log_win_prob(p2_elos, p1_elos)

    elo_season_divergence = elo_season_stability * jnp.mean((delos[:, 1:] - delos[:, :-1])**2)
    mean_log_data_prob = jnp.mean(winner_win_prob_log)
    return mean_log_data_prob - elo_season_divergence, jnp.exp2(mean_log_data_prob)

    # TODO: This is an experiment trying to evaluate ELO playing consistency. Try again and delete if does not work.
    # cons = params['consistency']
    # p1_cons = jnp.take(cons, p1s)
    # p2_cons = jnp.take(cons, p2s)
    # winner_win_prob_log = 0.0
    # winner_win_prob_log += p1_win_probs * log_win_prob_diff(diff/jnp.exp(p1_cons)) + p2_win_probs * log_win_prob_diff(-diff/jnp.exp(p1_cons))
    # winner_win_prob_log += p1_win_probs * log_win_prob_diff(diff/jnp.exp(p2_cons)) + p2_win_probs * log_win_prob_diff(-diff/jnp.exp(p2_cons))
    # winner_win_prob_log /= 2
    # return jnp.mean(winner_win_prob_log) - 0.005*jnp.mean(cons ** 2)

  # Optimize for these params:
  params = {
    'elos': jnp.zeros([player_count, season_count], dtype=jnp.float64),
    # 'consistency': jnp.zeros([player_count, season_count]),
  }

  # Momentum gradient descent with restarts
  m_lr = 1.0
  lr = learning_rate
  momentum = tree_map(jnp.zeros_like, params)
  last_params = params
  last_eval = -1
  last_grad = tree_map(jnp.zeros_like, params)
  last_reset_step = 0

  for i in range(steps or 9999):
    (eval, model_fit), grad = jax.value_and_grad(model,has_aux=True)(params)
    if do_log:
      elos = grad['elos']
      q=jnp.sum(params['elos'] == last_params['elos']) / params['elos'].size
      if i > 100 and q > 0.9:
        break
      print(f'Step {i:4}: eval: {jnp.exp2(eval)} lr={lr:7} grad={jnp.linalg.norm(elos)} {q}')
    if False:
      # Standard batch gradient descent algorithm works too.
      params = tree_map(lambda p, g: p + lr * g, params, grad)
    else:
      if eval < last_eval:
        if do_log: print(f'reset to {jnp.exp2(last_eval)}')
        lr /= 1.5
        if last_reset_step == i-1:
          lr /= 4
        last_reset_step = i
        momentum = tree_map(jnp.zeros_like, params)
        # momentum /= 2.
        params, eval, grad = last_params, last_eval, last_grad
      else:
        if (i - last_reset_step) % 12  == 0:
          lr *= 1.5
        last_params, last_eval, last_grad = params, eval, grad
      momentum = tree_map(lambda m, g: m_lr * m + g, momentum, grad)
      params = tree_map(lambda p, m: p + lr * m, params, momentum)
  return params, model_fit


def train_test(do_log=False, steps=60, lr=30):
  true_elos = jnp.array([[8.0, 4.0], [2.0, 3.0], [0.0, 0.0],])
  p1s = []
  p2s = []
  p1_win_probs = []
  seasons = []
  player_count, season_count = true_elos.shape
  for p1 in range(player_count):
    for p2 in range(player_count):
      for season in range(season_count):
        p1s.append(p1)
        p2s.append(p2)
        p1_win_prob = win_prob(true_elos[p1][season], true_elos[p2][season])
        p1_win_probs.append(p1_win_prob)
        seasons.append(season)

  test_data = {
    'p1s': jnp.array(p1s),
    'p2s': jnp.array(p2s),
    'p1_win_probs': jnp.array(p1_win_probs),
    'seasons': jnp.array(seasons),
  }
  results, _ = train(test_data, steps=steps, do_log=do_log, learning_rate=lr, elo_season_stability=0.0)
  delos = results['elos']
  delos = delos - jnp.min(delos, axis=0, keepdims=True)
  err = jnp.linalg.norm(delos - jnp.array(true_elos))
  assert err < 0.02, f'FAIL err={err:.2f}; results={results}'

  print('PASS')


def train_iglo(do_log=True, steps=None, lr=30, path='./iglo.json', regularization = 0.1):
  with open(path, 'r') as f:
    data = json.load(f)


  selector = np.array(data['win_types']) != 'not_played'

  players = data['players']
  player_count = len(players)
  season_count = 20

  first_season = [99999999] * player_count
  last_season = [-1] * player_count

  for p1, p2, s in zip(data['p1s'], data['p2s'], data['seasons']):
    first_season[p1] = min(first_season[p1], s)
    first_season[p2] = min(first_season[p2], s)
    last_season[p1] = max(last_season[p1], s)
    last_season[p2] = max(last_season[p2], s)

  data = {
    'p1s': jnp.array(data['p1s'])[selector],
    'p2s': jnp.array(data['p2s'])[selector],
    'p1_win_probs': jnp.array(data['p1_win_probs'])[selector],
    'seasons': jnp.array(data['seasons'])[selector],
  }
  data['p1_win_probs'] = (1-regularization) * data['p1_win_probs'] + regularization * 0.5

  params, model_fit = train(data, steps=steps, learning_rate=lr, do_log=do_log)
  delos = params['elos']
  assert delos.shape == (player_count, season_count), delos.shape

  # Sort by last season's elo
  results = sorted(zip(delos[:, -1], players, delos, first_season, last_season))
  results.reverse()

  for _, p, delos, first_season, last_season in results:
    delos = delos * 100 + 2000
    print(f'{p:18} ({first_season:2}-{last_season:2}): ', end='')
    for s in range(season_count):
      print(f'{delos[s]: 6.1f} ', end='') #  cons={jnp.exp(c)*100.0: 8.2f}')
    print()

  # expected_fit = 0.5758981704711914
  # expected_fit = 0.6161791524954028
  expected_fit = 0.6304865302054197  # without cross-season loss
  print(f'Model fit: {model_fit} improvement={model_fit-expected_fit}')
  return results


def show_plot(r):
  for _, pl, elo, first_season, last_season in r[:]:
    seasons = list(range(first_season, last_season+1))
    print(seasons)
    elo = elo[first_season:last_season+1]
    # print(list(seasons))
    plt.plot(seasons, elo*100+2000, label=pl)
  plt.legend()
  plt.show()


def train_show(last_pl=None, first_pl=None, steps=None):
  train_test()
  r = train_iglo(steps=steps)
  show_plot(r[first_pl:last_pl])