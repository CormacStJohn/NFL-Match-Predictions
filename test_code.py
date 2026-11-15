"""
aaubfiuebfiuebs
fsefsefs
"""

import nflreadpy as nfl

pbp = nfl.load_rosters()

print(pbp.filter(pbp["team"] == "ATL")["gsis_id"].head())
