#
import requests
import jax.numpy as jnp
import jax
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
  for season in request(f'seasons'):
    sn = season['number']
    if sn < 16: continue
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
            p1_win_probs.append(1.0 if winner_id == p1_id else 0.0)
            assert p1_id in id_to_player.keys()
            assert p2_id in id_to_player.keys()
            p1_name = id_to_player[p1_id]
            p2_name = id_to_player[p2_id]
            p1s.append(players.index(p1_name))
            p2s.append(players.index(p2_name))
  return {
    'players': players,
    'p1_win_probs': p1_win_probs,
    'p1s': p1s,
    'p2s': p2s,
  }

iglo_json_path = '/tmp/iglo.json'

def save_iglo_data():
  with open(iglo_json_path, 'w') as f:
    json.dump(get_data(), f)

def pow(x):
  return jnp.exp(np.log(2) * x)

def log(x):
  return jnp.log(x) / np.log(2)

def win_prob(p1_elo, p2_elo):
  return pow(p1_elo) / (pow(p1_elo) + pow(p2_elo))

def train(data):
  player_count = len(data['players'])
  # print(data['players'])
  # print(player_count)
  p1_win_probs = jnp.array(data['p1_win_probs'])
  p2_win_probs = 1.0 - p1_win_probs
  p1s = jnp.array(data['p1s'])
  p2s = jnp.array(data['p2s'])

  def model(elos):
    p1_elos = jnp.take(elos, p1s)
    p2_elos = jnp.take(elos, p2s)
    # print(p1_elos, p1s, elos)
    p1_win_prob_log = log(win_prob(p1_elos, p2_elos))
    p2_win_prob_log = log(win_prob(p2_elos, p1_elos))
    winner_win_prob_log = p1_win_probs * p1_win_prob_log + p2_win_probs * p2_win_prob_log
    return jnp.mean(winner_win_prob_log)

  elos = jnp.zeros([player_count])
  # print(model(elos))
  steps = 310
  lr = 200
  m_lr = 1.0

  momentum = jnp.zeros_like(elos)
  last_elos = jnp.zeros_like(elos)
  last_eval = -1
  last_grad = jnp.zeros_like(elos)
  for i in range(steps):
    eval, grad = jax.value_and_grad(model)(elos)
    # if (i+1) % (steps//10) == 0:
    print(f'Step {i:4}: eval: {pow(eval)}')
    if eval < last_eval:
      print(f'reset to {pow(last_eval)}')
      momentum = jnp.zeros_like(elos)
      elos, eval, grad = last_elos, last_eval, last_grad
    else:
      last_elos, last_eval, last_grad = elos, eval, grad
    momentum = m_lr * momentum + grad
    elos = elos + lr * momentum
  # return
  for elo, p in sorted(zip(elos, data['players'])):
    print(p, elo)
  print()

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
      print(p1, p2, p1_win_prob)
  test_data = {
    'players': { pi: f'elo{elos[pi]}' for pi in range(len(elos)) },
    'p1s': p1s,
    'p2s': p2s,
    'p1_win_probs': p1_win_probs,
  }

  train(test_data)

def train_iglo():
  with open(iglo_json_path, 'r') as f:
    data = json.load(f)
  train(data)