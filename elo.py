# ToDo:
# - player variance added
# - heavy tail fit added
# - per season gaussian process
# - Plan how to present
#   - present partial first.
#   - Short writup in markdown
#   - plot

import requests
import jax.numpy as jnp
import jax
from jax.tree_util import tree_map
import json
import numpy as np
# import aiohttp


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


def train(
  data,
  steps,
  do_log=False,
  learning_rate=30,
):
  player_count = len(data['players'])
  p1_win_probs = jnp.array(data['p1_win_probs'])
  p2_win_probs = 1.0 - p1_win_probs
  p1s = jnp.array(data['p1s'])
  p2s = jnp.array(data['p2s'])

  def model(params):
    elos = params['elos']
    cons = params['consistency']
    p1_elos = jnp.take(elos, p1s)
    p2_elos = jnp.take(elos, p2s)
    p1_cons = jnp.take(cons, p1s)
    p2_cons = jnp.take(cons, p2s)
    cons2 = jnp.exp((p1_cons + p2_cons) / 2)
    # p1_win_prob_log = log(win_prob(p1_elos, p2_elos))
    # p2_win_prob_log = log(win_prob(p2_elos, p1_elos))
    # winner_win_prob_log = p1_win_probs * p1_win_prob_log + p2_win_probs * p2_win_prob_log

    diff = (p2_elos-p1_elos)
    # diff = (p2_elos-p1_elos) / cons2
    winner_win_prob_log = p1_win_probs * log1pow(diff) + p2_win_probs * log1pow(-diff)

    return jnp.mean(winner_win_prob_log)
    # return jnp.mean(winner_win_prob_log) - 0.05*jnp.mean(cons ** 2)

  # Optimize for these params:
  params = {
    'elos': jnp.zeros([player_count]),
    'consistency': jnp.zeros([player_count]),
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
  return params, pow(last_eval)


def test_train():
  elos = [8.0, 2.0, 0.0]
  p1s = []
  p2s = []
  p1_win_probs = []
  for p1 in range(len(elos)):
    for p2 in range(len(elos)):
      p1s.append(p1)
      p2s.append(p2)
      p1_win_prob = win_prob(elos[p1], elos[p2])
      p1_win_probs.append(p1_win_prob)
      # print(p1, p2, p1_win_prob)
  test_data = {
    'players': { pi: f'elo{elos[pi]}' for pi in range(len(elos)) },
    'p1s': p1s,
    'p2s': p2s,
    'p1_win_probs': p1_win_probs,
  }

  results, _ = train(test_data, 30)
  results['elos'] = results['elos'] - jnp.min(results['elos'])
  err = jnp.linalg.norm(results['elos'] - jnp.array(elos))
  assert err < 0.02, f'FAIL err={err:.2f}; results={results}'
  print('PASS')


def train_iglo():
  regularization = 0.1
  with open(iglo_json_path, 'r') as f:
    data = json.load(f)

  for i in range(len(data['p1_win_probs'])):
    data['p1_win_probs'][i] = (1-regularization) * data['p1_win_probs'][i] + regularization * 0.5

  params, eval = train(data, steps=500, learning_rate=30, do_log=True)
  results = sorted(zip(params['elos'], data['players'], params['consistency']))
  results.reverse()


  for elo, p, c in results:
    print(f'{p:30}: {elo*100+2000: 8.2f}  cons={jnp.exp(c)*100.0: 8.2f}')
  expected_eval = 0.5758971571922302
  print(f'Model fit: {eval} Diff={eval-expected_eval}')

def main():
  test_train()
  train_iglo()
