breakout = [0, 1, 3, 4, 11, 12]
freeway = [0, 1, 5]
pong = [0, 1, 3, 4, 11, 12]
riverraid = range(18)
seaquest = range(18)
space_invaders = [0, 1, 3, 4, 11, 12]

def get_number_of_legal_actions(rom_name):
	if rom_name.startswith("breakout"):
		return len(breakout)
	elif rom_name.startswith("freeway"):
		return len(freeway)
	elif rom_name.startswith("pong"):
		return len(pong)
	elif rom_name.startswith("riverraid"):
		return len(riverraid)
	elif rom_name.startswith("seaquest"):
		return len(seaquest)
	elif rom_name.startswith("space_invaders"):
		return len(space_invaders)
	else:
		return 18

def map_action(action, rom_name):
	if rom_name.startswith("breakout"):
		return breakout[aciton]
	elif rom_name.startswith("freeway"):
		return freeway[action]
	elif rom_name.startswith("pong"):
		return pong[action]
	elif rom_name.startswith("riverraid"):
		return riverraid[action]
	elif rom_name.startswith("seaquest"):
		return seaquest[action]
	elif rom_name.startswith("space_invaders"):
		return space_invaders[action]
	else:
		return action
