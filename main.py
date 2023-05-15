# Copyright (C) 2023 zyxkad@gmail.com

import abc
import math
import random
from typing import Any

import pygame
import tensorflow

from ag import *

__all__ = [
	'main'
]

def random_chose(weights: list[tuple[Any, float]]) -> Any:
	assert len(weights) > 0
	total_weight = sum(w for _, w in weights)
	r = random.random() * total_weight
	i = 0
	for x, w in weights:
		i += w
		if r < i:
			return x
	raise RuntimeError('Unexpect statment')

class Food(Node):
	_cache = {}
	score_weights = [
		(1, 10.0),
		(2, 8.0),
		(3, 3.0),
		(4, 2.0),
		(5, 1.0),
	]

	def __init__(self, x: int, y: int, size: int, score: int, color: Color, isdeadbody: bool = False):
		super().__init__(x=x, y=y, width=size, height=size)
		self.isize = size
		self._score = score
		self.color = color
		self.isdeadbody = isdeadbody

	@classmethod
	def random(cls, width: int, height: int, color: Color):
		score = random_chose(cls.score_weights)
		size = round(10 * math.sqrt(score))
		return cls(
			round((random.random() - 0.5) * (width - size)),
			round((random.random() - 0.5) * (height - size)),
			size, score, color)

	@property
	def score(self) -> int:
		return self._score

	def on_draw(self, surface):
		cls = self.__class__
		ch = cls._cache.get((self.isize, self.color), None)
		if ch is None:
			ch = Surface((self.width, self.height))
			ch.polygon(self.color, [
				Vec2(self.isize / 4, self.isize / 2),
				Vec2(self.isize / 2, 0),
				Vec2(self.isize / 4 * 3, self.isize / 2),
				Vec2(self.isize / 2, self.isize),
			])
			cls._cache[(self.isize, self.color)] = ch
		surface.clone_from(ch)

class Snake(abc.ABC, Node):
	turn_speed = 200
	init_speed = 100
	init_nodes = 5
	speed_up_cost = 2

	def __init__(self, x: int, y: int, angle: int, name: str, body_color: Color):
		cls = self.__class__
		super().__init__()
		self.isize = 16
		self.hx = x
		self.hy = y
		self.speed = cls.init_speed
		self.angle = angle % 360
		self._score = 0
		self.kill_count = 0
		self._last_score = 0
		self._last_score2 = 50
		self.nodes = [(self.hx, self.hy, self.angle) for i in range(self.node_step * cls.init_nodes)]
		self._name = name
		self.body_color = body_color

	@property
	def hpos(self) -> Vec2:
		return Vec2(self.hx, self.hy)

	@property
	def score(self) -> int:
		return int(self._score)

	@property
	def node_step(self) -> int:
		return int(self.isize // 2 * 100 // self.speed * 0.85)

	@property
	def name(self) -> str:
		return self._name

	@property
	@abc.abstractmethod
	def left_active(self) -> bool:
		raise NotImplementedError()

	@property
	@abc.abstractmethod
	def right_active(self) -> bool:
		raise NotImplementedError()

	@property
	@abc.abstractmethod
	def speed_active(self) -> bool:
		raise NotImplementedError()

	def reset(self, x: int, y: int, angle: int):
		cls = self.__class__
		self.isize = 16
		self.hx = x
		self.hy = y
		self.speed = cls.init_speed
		self.angle = angle % 360
		self._score = 0
		self.kill_count = 0
		self._last_score = 0
		self._last_score2 = 50
		self.nodes = [(self.hx, self.hy, self.angle) for i in range(self.node_step * cls.init_nodes)]
		return self

	def move(self, dx: float, dy: float):
		self.hx += dx
		self.hy += dy

	def on_update(self, dt: float):
		speed = 1
		if self.speed_active:
			if self._score >= self.speed_up_cost * dt:
				self._score -= self.speed_up_cost * dt
				speed = 2
		dtn = int(dt * 20)
		dtf = 0.02
		step = self.speed * dtf
		for _ in range(dtn * speed):
			if self.left_active:
				self.angle = (self.angle + self.turn_speed * dtf) % 360
			if self.right_active:
				self.angle = (self.angle - self.turn_speed * dtf) % 360
			dx = -step * math.cos(self.angle / 180 * math.pi)
			dy = step * math.sin(self.angle / 180 * math.pi)
			self.move(dx, dy)
			self.nodes.insert(0, (self.hx, self.hy, self.angle))
			self.nodes.pop(-1)

		half_size = self.isize // 2
		if self._last_score + half_size < self.score:
			self._last_score += half_size
			if self.isize <= 50 and self._last_score2 * 2 < self.score:
				self._last_score2 = self._last_score2 * 2
				self.isize += 2
				self.speed += 2
			else:
				last_node = self.nodes[-1]
				self.nodes.extend(last_node for i in range(self.node_step))
		elif self._last_score - half_size > self.score:
			self._last_score -= half_size
			if self._last_score2 > 50 and self._last_score2 // 2 > self.score:
				self._last_score2 = self._last_score2 // 2
				self.isize -= 2
				self.speed -= 2
			else:
				self.nodes = self.nodes[:-self.node_step]

	def on_draw(self, surface):
		r = self.isize / 2
		head = Surface((self.isize, self.isize))
		head.circle(self.body_color, (r, r), r)
		surface.blit(head, (self.hx, self.hy))

		body = Surface((self.isize, self.isize))
		body.circle(self.body_color, (r, r), r)
		for i in range(self.node_step, len(self.nodes), self.node_step):
			x, y, a = self.nodes[i]
			surface.blit(body, (x, y))

		name_sf = Font.default(20).render(self.name, Colors.white, True)
		namebg_sf = Surface(name_sf.size + (6, 4))
		namebg_sf.fill(Color(0x00, 0x00, 0x00, 0x40))
		namebg_sf.blit(name_sf, (3, 3), anchor=Anchor.TOP_LEFT)
		surface.blit(namebg_sf, (self.hx, self.hy - 6), anchor=Anchor.BOTTOM_CENTER)

class Player(Snake):
	def __init__(self, x: int, y: int, angle: int, name: str, body_color: Color):
		super().__init__(x, y, angle, name, body_color)
		self.d = Director()
		self.d.camera.x = x
		self.d.camera.y = y
		self.camera_follow_x = 100
		self.camera_follow_y = 60

	@property
	def left_active(self) -> bool:
		return self.d.is_keydown(pygame.constants.K_a)

	@property
	def right_active(self) -> bool:
		return self.d.is_keydown(pygame.constants.K_d)

	@property
	def speed_active(self) -> bool:
		return self.d.is_keydown(pygame.constants.K_SPACE)

	def move(self, dx: int, dy: int):
		super().move(dx, dy)
		if abs(self.hx - self.d.camera.x) >= self.camera_follow_x:
			self.d.camera.x += dx
		if abs(self.hy - self.d.camera.y) >= self.camera_follow_y:
			self.d.camera.y += dy

	def reset(self, x: int, y: int, angle: int):
		self.d.camera.x = x
		self.d.camera.y = y
		return super().reset(x, y, angle)

######## BEGIN model ########

# class SnakeModel(Model):
#   def __init__(self):
#     super().__init__()
#     self.conv1 = Conv2D(32, 3, activation='relu')
#     self.flatten = Flatten()
#     self.d1 = Dense(128, activation='relu')
#     self.d2 = Dense(10)

#   def call(self, x):
#     x = self.conv1(x)
#     x = self.flatten(x)
#     x = self.d1(x)
#     return self.d2(x)

# # Create an instance of the model
# model = SnakeModel()

######## END model ########

class BotSnake(Snake):
	def __init__(self, x: int, y: int, angle: int, name: str, body_color: Color):
		super().__init__(x, y, angle, name, body_color)
		self._left_active = False
		self._right_active = not False
		self._speed_active = False

	@property
	def left_active(self) -> bool:
		return self._left_active or random.randint(0, 1) == 0

	@property
	def right_active(self) -> bool:
		return self._right_active

	@property
	def speed_active(self) -> bool:
		return self._speed_active or random.randint(0, 1) == 0

class PlayLayer(Layer):
	def __init__(self, bwidth: int, bheight: int, size: int):
		super().__init__()
		self.d = Director()
		self.bwidth = bwidth
		self.bheight = bheight
		self.isize = size
		self.broad_color = Color.from_rgb(0.9, 0.9, 0.9)
		self.broad_line_color = Color.from_rgb(0.5, 0.5, 0.5)

		self.snakes = []
		rp = self.random_pos()
		self.player = Player(int(rp.x), int(rp.y), int(random.random() * 4) * 90, 'Player', Colors.blue)
		self.add_snake(self.player)
		for i in range(9):
			rp = self.random_pos()
			s = BotSnake(
				int(rp.x), int(rp.y),
				int(random.random() * 360),
				f'Bot_{i}',
				Color(random.randint(0, 0xff), random.randint(0, 0xff), random.randint(0, 0xff)))
			self.add_snake(s)

		self.foods = []
		init_food_count = self.bwidth * self.bheight // 5 # 1/5 foods per square
		for _ in range(init_food_count):
			self.make_food()

	@property
	def width(self) -> float:
		return self.bwidth * self.isize

	@property
	def height(self) -> float:
		return self.bheight * self.isize

	def random_pos(self) -> Vec2:
		return Vec2(
			(random.random() - 0.5) * (self.width - self.isize * 6),
			(random.random() - 0.5) * (self.height - self.isize * 6))

	def add_snake(self, s: Snake):
		self.snakes.append(s)
		self.add_child(s, z_index=10)

	def remove_snake(self, s: Snake):
		self.snakes.remove(s)
		self.remove_child(s)

	def add_food(self, f: Food):
		self.add_child(f)
		self.foods.append(f)

	def remove_food(self, f: Food):
		self.foods.remove(f)
		self.remove_child(f)
		if not f.isdeadbody:
			self.make_food()

	def on_player_dead(self):
		def retrieve():
			rp = self.random_pos()
			self.add_snake(self.player.reset(
				int(rp.x), int(rp.y),
				int(random.random() * 8) * 45))
		self.scheduler.add_timeout(retrieve, 3)

	def on_bot_dead(self, s: Snake):
		def retrieve():
			rp = self.random_pos()
			self.add_snake(s.reset(
				int(rp.x), int(rp.y),
				int(random.random() * 360)))
		self.scheduler.add_timeout(retrieve, 3)

	def on_update(self, dt: float):
		snakes = self.snakes.copy()
		for s in snakes:
			s.on_update(dt)
			spos = s.hpos
			for f in self.foods.copy():
				if spos.in_range(f.pos, (self.isize + f.isize) / 2):
					s._score += f.score
					self.remove_food(f)
			dead = s.hx < (-self.width + s.isize) // 2 or \
				 s.hx > (self.width - s.isize) // 2 or \
				 s.hy < (-self.height + s.isize) // 2 or \
				 s.hy > (self.height - s.isize) // 2
			for s2 in snakes:
				if dead:
					break
				if s2 is not s:
					for x, y, _ in s2.nodes[:-s2.node_step]:
						if spos.in_range(Vec2(x, y), (s.isize + s2.isize) // 2):
							s2.kill_count += 1
							dead = True
							break
			if dead:
				score_remain = s.score + 10
				node_per_score = 10
				avg_score = score_remain * node_per_score // len(s.nodes)

				for x, y, _ in (s.nodes[i] for i in range(0, len(s.nodes), node_per_score)):
					self.add_food(Food(
						x + (random.random() - 0.5) * 10,
						y + (random.random() - 0.5) * 10,
						max(math.sqrt(avg_score), s.isize), avg_score, s.body_color,
						isdeadbody=True))
				self.remove_snake(s)
				if s is self.player:
					self.on_player_dead()
				else:
					self.on_bot_dead(s)

	def on_draw(self, surface):
		surface.fill(self.broad_color)
		for x in range(1, self.bwidth):
			surface.fill(self.broad_line_color, Rect(x * self.isize, 0, 1, self.height))
		for y in range(1, self.bheight):
			surface.fill(self.broad_line_color, Rect(0, y * self.isize, self.width, 1))

	def make_food(self):
		f = Food.random(self.width, self.height, Colors.yellow)
		self.add_food(f)

class GUILayer(UILayer):
	def __init__(self):
		super().__init__()
		self.d = Director()
		self.font = Font.default(20)
		self.score = 0
		self.killed = 0
		self.ranks = []

	def on_draw(self, surface):
		fps = self.d.real_fps
		left_box = Surface((100, 70))
		left_box.fill(Color(0, 0, 0, 0x50))
		left_box.blit(
			self.font.render(f'FPS: {fps:.2f}', Colors.white, True),
			Vec2(5, 5), anchor=Anchor.TOP_LEFT)
		left_box.blit(
			self.font.render(f'Score: {self.score:d}', Colors.white, True),
			Vec2(5, 25), anchor=Anchor.TOP_LEFT)
		left_box.blit(
			self.font.render(f'Killed: {self.killed:d}', Colors.white, True),
			Vec2(5, 45), anchor=Anchor.TOP_LEFT)
		surface.blit(left_box, (0, 0), anchor=Anchor.TOP_LEFT)

		right_box = Surface((200, len(self.ranks) * 20 + 5))
		right_box.fill(Color(0, 0, 0, 0x80))
		for i, (n, s, k) in enumerate(self.ranks):
			right_box.blit(
				self.font.render(f'{i + 1} {n:12s} {s:-7d} {k:-4d}', Colors.white, True),
				Vec2(5, 5 + i * 20), anchor=Anchor.TOP_LEFT)
		surface.blit(right_box, (surface.size.x, 0), anchor=Anchor.TOP_RIGHT)

class PlayScene(Scene):
	def __init__(self):
		plyl = PlayLayer(50, 50, 20)
		gui = GUILayer()
		super().__init__(plyl, gui)
		self.plyl = plyl
		self.gui = gui
		self.wall_color = Color(0xff, 0x40, 0x50)

	def on_entered(self):
		super().on_entered()
		self.schedule_update()

	def on_update(self, dt: float):
		self.gui.score = self.plyl.player.score
		self.gui.killed = self.plyl.player.kill_count
		self.plyl.on_update(dt)

		self.gui.ranks = [('<unknown>', 0, 0) for _ in range(5)]
		for i, s in enumerate(list(sorted(self.plyl.snakes, key=lambda s: s.score, reverse=True))[:5]):
			self.gui.ranks[i] = (s.name, s.score, s.kill_count)

	def on_draw(self, surface):
		surface.fill(self.wall_color)

def main():
	d = Director()
	d.init_with_window((1200, 700), 'AG')
	winsize = d.winsize
	ps = PlayScene()
	d.fps = 60
	d.run_with_scene(ps)

if __name__ == '__main__':
	main()
