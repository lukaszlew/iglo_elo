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

from jax.config import config
config.update("jax_numpy_rank_promotion", "raise")
config.update("jax_enable_x64", True)




def win_prob(p1_elo, p2_elo):
  return 1.0 / (1.0 + jnp.exp2(p2_elo-p1_elo))
  # This is more understandable and equivalent:
  # return jnp.exp2(p1_elo) / (jnp.exp2(p1_elo) + jnp.exp2(p2_elo))

def train(
  data,
  steps,
  do_log=False,
  learning_rate=10000,
  elo_stability=0.5,
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

    # p1_win_prob_log = jnp.log2(win_prob(p1_elos, p2_elos))
    # p2_win_prob_log = jnp.log2(win_prob(p2_elos, p1_elos))
    # winner_win_prob_log = p1_win_probs * p1_win_prob_log + p2_win_probs * p2_win_prob_log

    diff = (p2_elos-p1_elos)
    def log1pow(x):
      return -jnp.log2(1.0 + jnp.exp2(x))
    winner_win_prob_log = p1_win_probs * log1pow(diff) + p2_win_probs * log1pow(-diff)
    elo_divergence = jnp.mean((delos[:, 1:] - delos[:, :-1])**2)
    eval = jnp.mean(winner_win_prob_log)
    return eval - elo_stability * elo_divergence, eval

    # return jnp.mean(winner_win_prob_log) - 0.01*jnp.mean(delos**2)
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

  print(params['elos'].dtype)
  for i in range(steps):
    (eval, eval1), grad = jax.value_and_grad(model,has_aux=True)(params)
    if do_log:
      elos = grad['elos']
      q=jnp.sum(params['elos'] == last_params['elos'])
      print(f'Step {i:4}: eval: {jnp.exp2(eval)} lr={lr:7} grad={jnp.linalg.norm(elos)} {q}')
      # print(eval - last_eval)
    # lr = learning_rate * (steps-i) / steps
    # lr = learning_rate / jnp.sqrt(i+1)
    # lr = learning_rate / (i+1)
    if False:
      # Batch gradient descent algorithm.
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
      # params['consistency'] = jnp.zeros_like(params['consistency'])
      # params['consistency'] -= jnp.mean(params['consistency'])
      # params['consistency'] /= 2
      # params['consistency'] = jnp.ones_like(params['consistency'])
  return params, jnp.exp2(eval1)


def test1(do_log=False, steps=30, lr=30):
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
        # print(p1, p2, p1_win_prob)
  # players = { pi: f'elo{true_elos[pi]}' for pi in range(player_count) }

  test_data = {
    'p1s': jnp.array(p1s),
    'p2s': jnp.array(p2s),
    'p1_win_probs': jnp.array(p1_win_probs),
    'seasons': jnp.array(seasons),
  }
  results, _ = train(test_data, steps=steps, do_log=do_log, learning_rate=lr)
  delos = results['elos']
  delos = delos - jnp.min(delos, axis=0, keepdims=True)
  err = jnp.linalg.norm(delos - jnp.array(true_elos))
  assert err < 0.02, f'FAIL err={err:.2f}; results={results}'

  print('PASS')


def iglo(do_log=True, steps=650, lr=30, path='./iglo.json'):
  regularization = 0.1
  with open(path, 'r') as f:
    data = json.load(f)

  # print(data.keys())
  # print(set(data['win_types']))
  # return

  selector = np.array(data['win_types']) != 'not_played'

  players = data['players']
  data = {
    'p1s': jnp.array(data['p1s'])[selector],
    'p2s': jnp.array(data['p2s'])[selector],
    'p1_win_probs': jnp.array(data['p1_win_probs'])[selector],
    'seasons': jnp.array(data['seasons'])[selector],
  }
  data['p1_win_probs'] = (1-regularization) * data['p1_win_probs'] + regularization * 0.5

  params, eval = train(data, steps=steps, learning_rate=lr, do_log=do_log)
  player_count, season_count = 187, 20
  delos = params['elos']
  assert delos.shape == (player_count, season_count), delos.shape


  # results = sorted(zip(delos, players, params['consistency']))
  # print(delos)
  results = sorted(zip(delos[:, -1], players, delos))
  results.reverse()

  for elo, p, delos in results:
    print(f'{p:30}: ', end='')
    for s in range(season_count):
      print(f'{delos[s]: 6.2f} ', end='') #  cons={jnp.exp(c)*100.0: 8.2f}')
      # print(f'{delos[s]*100+2000: 6.0f} ', end='') #  cons={jnp.exp(c)*100.0: 8.2f}')
    print()

  # expected_eval = 0.5758981704711914
  expected_eval = 0.6161791524954028
  print(f'Model fit: {eval} improvement={eval-expected_eval}')
  return results

import matplotlib.pyplot as plt

def p(r):
  for _, pl, elo in r[:]:
    plt.plot(range(20), elo*100+2000, label=pl)
  plt.legend()
  plt.show()

def main():
  test_train()
