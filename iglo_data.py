import requests
import json

# TODO: Implement faster fetching using aiohttp parallelism
# import aiohttp

url = 'https://iglo.szalenisamuraje.org/api'

def request(s):
  print(f'getting {s}')
  return requests.get(f'{url}/{s}').json()['results']


def get_data(test=False):
  players = []
  p1_win_probs = []
  p1s = []
  p2s = []
  seasons = []
  win_types = []
  groups = []
  for season in request(f'seasons'):
    sn = season['number']
    if test:
      if sn < 18: continue
    for group in request(f'seasons/{sn}/groups'):
      gn = group['name']
      if test:
          if group['name'] > 'B': continue
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
            win_types.append(win_type)
            groups.append(gn)
  return {
    'players': players,
    'p1_win_probs': p1_win_probs,
    'p1s': p1s,
    'p2s': p2s,
    'seasons': seasons,
    'win_types': win_types,
    'groups': groups
  }


def save_iglo_data(path = './iglo.json'):
  with open(path, 'w') as f:
    json.dump(get_data(False), f)
