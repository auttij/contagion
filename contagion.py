import pygame
import numpy as np
import sys
from noise import pnoise2
from itertools import product
import random
import time

# some global values
ON = "#"
OFF = " "
BLACK = "#000000"
WHITE = "#FFFFFF"

# global variables
DEBUG = 0
FPS = 144
THRESHOLD=0.505
SCALE = 15
OCTAVES = 1
PERSISTENCE = 0.01
LACUNARITY = 2.0
SEED = None

cs = 1
array_width = 720
array_height = 360
screen_width = array_width * (cs*2) + 2 * (cs*2)
screen_height = array_height * (cs*2) + 2 * (cs*2)

class Contagion:
	def __init__(self, w, h, screen):
		self.cells = []
		self.screen = screen
		self.w = w
		self.h = h
		self.array = []

		self.new_landscape()
		self.add_neighbors()

	def reset_screen(self):
		self.screen.fill(WHITE)
		pygame.display.update()

	def draw_cells(self):
		for cell in c.cells:
			c.draw_update(cell)
		pygame.display.update()

	def draw_update(self, cell):
		pygame.draw.circle(self.screen, cell.color, (cell.sx, cell.sy), cs, 0)

	def draw_arr(self):
		pass

	def print_array(self):
		for r in self.array:
			for v in r:
				print(v, end="")
			print()

	def new_landscape(self, threshold = 0.505, scale = 15,  octaves = 1, persistence = 0.01, lacunarity = 2.0, seed = None):
		self.cells = []
		self.array = self.create_landscape(shape=(self.h, self.w), threshold=threshold, scale=scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, seed=seed)
		self.add_neighbors()

	def create_landscape(self, shape = (200, 200), threshold = 0.505, scale = 15,  octaves = 1, persistence = 0.01, lacunarity = 2.0, seed = None):
		if DEBUG == 1:
			start_time = time.time()
		parr = perlin_array(shape=shape, scale=scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, seed=seed)
		if DEBUG == 1:
			ptime = time.time()
		arr = [[0 for i in parr[0]] for i in parr]
		for y, line in enumerate(parr):
			for x, val in enumerate(line):
				if val > threshold:
					cell = Cell(ON, BLACK, x, y)
					self.cells.append(cell)
					arr[y][x] = cell
				else:
					arr[y][x] = Cell(OFF, WHITE, x, y)
		if DEBUG == 1:
			atime = time.time()
			print(f"perl noise in ${ptime-start_time}s, array handling in ${atime-ptime}s")
		return arr

	def add_neighbors(self):
		for y, i in enumerate(self.array):
			for x, ii in enumerate(i):
				if self.array[y][x].power == ON:
					self.array[y][x].neighbors = [i for i in (self.array[ny][nx] for nx, ny in neighbors(x, y, self.w, self.h)) if i.power == ON]

	def cycle_color(self, previous):
		colors = ["#AA7739", "#28774F", "#28536C", "#AA5439"]
		if previous in colors:
			return colors[(colors.index(previous) + 1) % len(colors)]
		else:
			return colors[0]

	def random_black_cell(self):
		if len(self.cells) > 0:
			return random.choice(self.cells)
		else:
			return None

	def random_on_cell(self):
		arr = self.array
		x, y = None, None
		while (x == None and y == None) or arr[y][x].power == OFF:
			y = random.choice(range(len(self.array)))
			x = random.choice(range(len(self.array[0])))
		return arr[y][x]

	def create_waves(self, color):
		# start = self.random_on_cell()
		start = self.random_black_cell()
		if start == None:
			return []
		waves = [[start]]
		stack = [start]
		next_stack = [start]
		start.color = color
		seen =  [start]

		while len(next_stack) > 0:
			next_stack = []
			while len(stack) > 0:
				cell = stack.pop()
				nei = cell.neighbors
				for n in nei:
					if n.color != color:
						n.color = color
						next_stack.append(n)
			waves.append(next_stack[:])
			stack = next_stack
			seen += stack
		self.cells = list(set(self.cells) - set(seen))
		return waves

class Cell:
	def __init__(self, power, color, x, y):
		self.x = x
		self.y = y
		self.sx = x * (cs*2) + (cs*2)
		self.sy = y * (cs*2) + (cs*2)
		self.power = power
		self.color = color
		self.neighbors = []

	def colors(self):
		return (int(self.color[1:3], 16), int(self.color[3:5], 16), int(self.color[5:7], 16))

	def __str__(self):
		return self.power

	def __repr__(self):
		return self.__str__()

def add(input):
	a, b = input
	return (a[0]+b[0], a[1]+b[1])

def neighbors(x, y, xn, yn):
	# l = list(map(add, [((x, y), i) for i in [(-1,0), (1,0), (0,-1), (0,1)]]))
	l = list(map(add, [((x, y), i) for i in product(range(-1, 2), repeat=2)]))
	return [(nx, ny) for nx, ny in l if 0 <= nx and nx < xn and 0 <= ny and ny < yn and not (nx == x and ny == y)]

def perlin_array(shape = (200, 200),
			scale=100, octaves = 6, 
			persistence = 0.5, 
			lacunarity = 2.0, 
			seed = None):

		if not seed:
			seed = np.random.randint(100, 200)

		arr = np.zeros(shape)
		for i in range(shape[0]):
			for j in range(shape[1]):
				val = pnoise2(i / scale,
											j / scale,
											octaves=octaves,
											persistence=persistence,
											lacunarity=lacunarity,
											repeatx=1024,
											repeaty=1024,
											base=seed)
				arr[i][j] = val
		max_arr = np.max(arr)
		min_arr = np.min(arr)
		norm_me = lambda x: (x-min_arr)/(max_arr - min_arr)
		norm_me = np.vectorize(norm_me)
		arr = norm_me(arr)
		return arr

def update_width(w):
	global array_width, screen_width
	array_width = w
	screen_width = array_width * (cs*2) + 2 * (cs*2)

def update_height(h):
	global array_height, screen_height
	array_height = h
	screen_height = array_height * (cs*2) + 2 * (cs*2)

def check_cmd_arguments(**kwargs):
	for k, v in kwargs.items():
		if k.lower() == "debug":
			global DEBUG
			DEBUG = int(v == 'true')
		elif k.lower() == "w" or k.lower() == "width":
			update_width(int(v))
		elif k.lower() == "h" or k.lower() == "height":
			update_height(int(v))
		elif k.lower() == "fps":
			global FPS
			FPS = int(v)
		elif k.lower() == "threshold":
			global THRESHOLD
			THRESHOLD = float(v)
		elif k.lower() == "scale":
			global SCALE
			SCALE = int(v)
		elif k.lower() == "octaves":
			global OCTAVES
			OCTAVES = int(v)
		elif k.lower() == "persistence":
			global PERSISTENCE
			PERSISTENCE = float(v)
		elif k.lower() == "lacunarity":
			global LACUNARITY
			LACUNARITY = float(v)
		elif k.lower() == "seed":
			global SEED
			SEED = int(v)


def main(**kwargs):
	check_cmd_arguments(**kwargs)

	pygame.init()
	clock = pygame.time.Clock()

	screen = pygame.display.set_mode((screen_width,screen_height))
	c = Contagion(array_width, array_height, screen)
	color = WHITE
	c.reset_screen()

	waves = []
	loops = 5

	while True:
		msElapsed = clock.tick(FPS)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
					pygame.quit(); sys.exit()
		if len(waves) == 0:
			if len(c.cells) == 0:
				global THRESHOLD, SCALE, OCTAVES, PERSISTENCE, LACUNARITY, SEED
				c.new_landscape(THRESHOLD, SCALE, OCTAVES, PERSISTENCE, LACUNARITY, SEED)

			if DEBUG == 1:
				start_time = time.time()
			waves = []
			color = c.cycle_color(color)
			new_waves = c.create_waves(color)
			while len(new_waves) > 0:
				for i, wave in enumerate(new_waves):
					if len(waves) > i:
						waves[i] += wave
					else:
						waves.append(wave)

				color = c.cycle_color(color)
				new_waves = c.create_waves(color)
			waves = waves[::-1]
			
			if DEBUG == 1:
				end_time = time.time()
				print(f"wave calculation in ${end_time-start_time}s")#, wave times: ${wt}")
			c.reset_screen()
		else:
			wave = waves.pop()
			for cell in wave:
				# cell.color = color
				c.draw_update(cell)
		pygame.display.update()

if __name__ == "__main__":
	main(**dict(arg.split('=') for arg in sys.argv[1:]))

