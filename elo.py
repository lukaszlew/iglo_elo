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


def log_data_probability(p1_elos, p2_elos, p1_win_probs, p2_win_probs):
  winner_win_prob_log = p1_win_probs * log_win_prob(p1_elos, p2_elos) + p2_win_probs * log_win_prob(p2_elos, p1_elos)
  mean_log_data_prob = jnp.mean(winner_win_prob_log)
  return mean_log_data_prob


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
    elos = params['elos']
    assert elos.shape == (player_count, season_count)
    p1_elos = elos[p1s, seasons]
    p2_elos = elos[p2s, seasons]

    assert p1_elos.shape == (data_size,)
    assert p2_elos.shape == (data_size,)
    mean_log_data_prob = log_data_probability(p1_elos, p2_elos, p1_win_probs, p2_win_probs)
    elo_season_divergence = elo_season_stability * jnp.mean((elos[:, 1:] - elos[:, :-1])**2)
    geomean_data_prob = jnp.exp2(mean_log_data_prob)
    return mean_log_data_prob - elo_season_divergence, geomean_data_prob

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
  elos = results['elos']
  elos = elos - jnp.min(elos, axis=0, keepdims=True)
  err = jnp.linalg.norm(elos - jnp.array(true_elos))
  assert err < 0.02, f'FAIL err={err:.2f}; results={results}'

  print('PASS')


def train_iglo(do_log=True, steps=None, lr=30, path='./iglo.json', regularization = 0.1, save_json=False):
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
  elos = params['elos']
  assert elos.shape == (player_count, season_count), elos.shape


  # Sort by last season's elo
  order = jnp.flip(elos[:, -1].argsort())

  players = np.array(players)[order]
  elos = elos[order]
  first_season = jnp.array(first_season)[order]
  last_season = jnp.array(last_season)[order]

  for i in range(len(players)):
    p = players[i]
    e = elos[i] * 100 + 2000
    fs = first_season[i]
    ls = last_season[i]
    print(f'{p:18} ({fs:2}-{ls:2}): ', end='')
    for s in range(season_count):
      print(f'{e[s]: 6.1f} ', end='') #  cons={jnp.exp(c)*100.0: 8.2f}')
    print()

  # expected_fit = 0.5758981704711914
  # expected_fit = 0.6161791524954028
  # expected_fit = 0.6304865302054197  # without cross-season loss
  expected_fit = 0.6304865296890099
  print(f'Model fit: {model_fit} improvement={model_fit-expected_fit}')
  # This is the format of JSON export.
  # All lists are of the same length equal to the number of players.
  result = {
    'players': players.tolist(),
    # elos is a list of lists. For each player, we have ELO strength for a given season.
    'elos': elos.tolist(),
    'first_season': first_season.tolist(),
    'last_season': last_season.tolist(),
  }
  if save_json:
    with open('./iglo_elo.json', 'w') as f:
      json.dump(result, f)
  return result


def show_plot(pl_count, first_pl=0, save_svg=None):
  iglo_elo = read_iglo_elo()
  pl_count = pl_count or len(iglo_elo['players'])
  for i in range(first_pl, first_pl+pl_count):
    pl = iglo_elo['players'][i]
    elo = iglo_elo['elos'][i]
    fs = iglo_elo['first_season'][i]
    ls = iglo_elo['last_season'][i]
    seasons = list(range(fs, ls+1))
    elo = np.array(elo[fs:ls+1])
    plt.plot(seasons, elo*100+2000, label=pl, marker='.')
  plt.legend()
  if save_svg is not None:
    plt.savefig(save_svg)
  else:
    plt.show()


def read_iglo_elo():
  with open('./iglo_elo.json', 'r') as f:
    return json.load(f)


def show_elo_evolution_histogram(bins=100):
  elos = np.array(read_iglo_elo()['elos'])
  elo_deltas = elos[:, 1:] - elos[:, :-1]
  elo_deltas = elo_deltas.reshape(elo_deltas.size)
  elo_deltas = elo_deltas[np.abs(elo_deltas) > 0.00001]
  # elo_deltas = elo_deltas[np.abs(elo_deltas) < 0.2]
  # print(jnp.mean(elo_deltas))
  plt.hist(elo_deltas, bins=bins)
  plt.show()