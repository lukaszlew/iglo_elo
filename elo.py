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

import requests
import jax.numpy as jnp
import jax
from jax.tree_util import tree_map
import json
import numpy as np
# import aiohttp

from jax.config import config
config.update("jax_numpy_rank_promotion", "raise")


url = 'https://iglo.szalenisamuraje.org/api'


def request(s):
  print(f'getting {s}')
  return requests.get(f'{url}/{s}').json()['results']


def get_data():
  players = []
  p1_win_probs = []
  p1s = []
  p2s = []
  seasons = []
  for season in request(f'seasons'):
    sn = season['number']
    # if sn < 16: continue
    for group in request(f'seasons/{sn}/groups'):
      gn = group['name']
      # if group['name'] > 'B': continue
      id_to_player = {}
      for member in request(f'seasons/{sn}/groups/{gn}/members'):
        mid = member['id']
        mname = member['player']
        if mid not in id_to_player:
          id_to_player[mid] = mname
        if mname not in players:
          players.append(mname)
        assert id_to_player[mid] == mname, (member, players)

      for rounds in request(f'seasons/{sn}/groups/{gn}/rounds'):
        rn = rounds['number']
        for game in request(f'seasons/{sn}/groups/{gn}/rounds/{rn}/games'):
          p1_id = game['black']
          p2_id = game['white']
          winner_id = game['winner']
          win_type = game['win_type']
          if win_type == 'bye':
            continue
          assert winner_id is None or (winner_id == p1_id) or (winner_id == p2_id), (winner_id, p1_id, p2_id)
          if winner_id is not None:
            assert p1_id in id_to_player.keys()
            assert p2_id in id_to_player.keys()
            p1_name = id_to_player[p1_id]
            p2_name = id_to_player[p2_id]
            p1_win_probs.append(1.0 if winner_id == p1_id else 0.0)
            p1s.append(players.index(p1_name))
            p2s.append(players.index(p2_name))
            seasons.append(sn)
  return {
    'players': players,
    'p1_win_probs': p1_win_probs,
    'p1s': p1s,
    'p2s': p2s,
    'seasons': seasons,
  }


iglo_json_path = '/tmp/iglo.json'


def save_iglo_data():
  with open(iglo_json_path, 'w') as f:
    json.dump(get_data(), f)


def pow(x):
  return jnp.exp(np.log(2) * x)


def log(x):
  return jnp.log(x) / np.log(2)

def log1pow(x):
  return -log(1.0 + pow(x))

def win_prob(p1_elo, p2_elo):
  return pow(p1_elo) / (pow(p1_elo) + pow(p2_elo))
  # return 1.0 / (1.0 + pow(p2_elo-p1_elo))

# momentum restarting - we can do a mapping between loss component and parameter subspace
# restart only the subspace when the component increases

def train(
  data,
  steps,
  do_log=False,
  learning_rate=30,
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

    # p1_win_prob_log = log(win_prob(p1_elos, p2_elos))
    # p2_win_prob_log = log(win_prob(p2_elos, p1_elos))
    # winner_win_prob_log = p1_win_probs * p1_win_prob_log + p2_win_probs * p2_win_prob_log

    diff = (p2_elos-p1_elos)
    winner_win_prob_log = p1_win_probs * log1pow(diff) + p2_win_probs * log1pow(-diff)
    return jnp.mean(winner_win_prob_log)
    # delos = jnp.mean((elos[:,1:] - elos[:, :-1]) ** 2)
    # return jnp.mean(winner_win_prob_log) - 1.1 * delos

    # cons = params['consistency']
    # p1_cons = jnp.take(cons, p1s)
    # p2_cons = jnp.take(cons, p2s)
    # winner_win_prob_log = 0.0
    # winner_win_prob_log += p1_win_probs * log1pow(diff/jnp.exp(p1_cons)) + p2_win_probs * log1pow(-diff/jnp.exp(p1_cons))
    # winner_win_prob_log += p1_win_probs * log1pow(diff/jnp.exp(p2_cons)) + p2_win_probs * log1pow(-diff/jnp.exp(p2_cons))
    # winner_win_prob_log /= 2
    # return jnp.mean(winner_win_prob_log) - 0.005*jnp.mean(cons ** 2)

  # Optimize for these params:
  params = {
    'elos': jnp.zeros([player_count, season_count]),
    # 'consistency': jnp.zeros([player_count, season_count]),
  }
  if False:
    # Batch gradient descent algorithm.
    for i in range(steps):
      eval, grad = jax.value_and_grad(model)(params)
      if do_log: print(f'Step {i:4}: eval: {pow(eval)}')
      params = tree_map(lambda p, g: p + learning_rate * g, params, grad)
  else:
    # Momentum gradient descent with restarts
    m_lr = 1.0
    momentum = tree_map(jnp.zeros_like, params)
    last_params = params
    last_eval = -1
    last_grad = tree_map(jnp.zeros_like, params)

    for i in range(steps):
      eval, grad = jax.value_and_grad(model)(params)
      if do_log: print(f'Step {i:4}: eval: {pow(eval)}')
      if eval < last_eval:
        if do_log: print(f'reset to {pow(last_eval)}')
        momentum = tree_map(jnp.zeros_like, params)
        # momentum /= 2.
        params, eval, grad = last_params, last_eval, last_grad
      else:
        last_params, last_eval, last_grad = params, eval, grad
      momentum = tree_map(lambda m, g: m_lr * m + g, momentum, grad)
      params = tree_map(lambda p, m: p + learning_rate * m, params, momentum)
      # params['consistency'] = jnp.zeros_like(params['consistency'])
      # params['consistency'] -= jnp.mean(params['consistency'])
      # params['consistency'] /= 2
      # params['consistency'] = jnp.ones_like(params['consistency'])
  return params, pow(eval)


def test1(do_log=False):
  elos = jnp.array([[8.0, 4.0], [2.0, 3.0], [0.0, 0.0],])
  p1s = []
  p2s = []
  p1_win_probs = []
  seasons = []
  for p1 in range(len(elos)):
    for p2 in range(len(elos)):
      for season in range(2):
        p1s.append(p1)
        p2s.append(p2)
        p1_win_prob = win_prob(elos[p1][season], elos[p2][season])
        p1_win_probs.append(p1_win_prob)
        seasons.append(season)
        # print(p1, p2, p1_win_prob)
  # players = { pi: f'elo{elos[pi]}' for pi in range(len(elos)) }

  test_data = {
    'p1s': jnp.array(p1s),
    'p2s': jnp.array(p2s),
    'p1_win_probs': jnp.array(p1_win_probs),
    'seasons': jnp.array(seasons),
  }
  results, _ = train(test_data, 30, do_log=do_log)
  results['elos'] = results['elos'] - jnp.min(results['elos'], axis=0, keepdims=True)
  err = jnp.linalg.norm(results['elos'] - jnp.array(elos))
  assert err < 0.02, f'FAIL err={err:.2f}; results={results}'
  print('PASS')


def iglo():
  regularization = 0.1
  with open(iglo_json_path, 'r') as f:
    data = json.load(f)

  players = data['players']
  data = {
    'p1s': jnp.array(data['p1s']),
    'p2s': jnp.array(data['p2s']),
    'p1_win_probs': jnp.array(data['p1_win_probs']),
    'seasons': jnp.array(data['seasons']),
  }

  data['p1_win_probs'] = (1-regularization) * data['p1_win_probs'] + regularization * 0.5

  params, eval = train(data, steps=1000, learning_rate=30, do_log=True)
  player_count, season_count = 187, 20
  assert params['elos'].shape == (player_count, season_count), params['elos'].shape


  # results = sorted(zip(params['elos'], players, params['consistency']))
  # triu = jnp.triu(jnp.ones([season_count, season_count]))
  # print(params['elos'])
  # elos = params['elos'] @ triu
  elos = params['elos']
  # print(elos)
  results = sorted(zip(elos[:, -1], players, elos))
  results.reverse()

  for elo, p, elos in results:
    print(f'{p:30}: ', end='')
    for s in range(season_count):
      print(f'{elos[s]: 6.2f} ', end='') #  cons={jnp.exp(c)*100.0: 8.2f}')
      # print(f'{elos[s]*100+2000: 6.0f} ', end='') #  cons={jnp.exp(c)*100.0: 8.2f}')
    print()

  expected_eval = 0.5758981704711914
  print(f'Model fit: {eval} Diff={eval-expected_eval}')

def main():
  test_train()
